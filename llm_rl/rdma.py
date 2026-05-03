import os
import socket
import ray
import torch

NVSHMEM_ENV = {

    "NVSHMEM_REMOTE_TRANSPORT": "ibrc",
    "NVSHMEM_IB_ENABLE_IBGDA": "1",
    "NVSHMEM_IBGDA_NIC_HANDLER": "gpu",
    "NVSHMEM_IBGDA_NUM_RC_PER_PE": "1",
    "NVSHMEM_DISABLE_P2P": "0",
    "NVSHMEM_SYMMETRIC_SIZE": "1G",
    "NVSHMEM_BOOTSTRAP": "UID",
    "NVSHMEM_DEBUG": "INFO",
    "NVSHMEM_INFO": "1",
}


@ray.remote(num_gpus=1)
class RdmaWorker:
    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size

    def host_info(self):
        """Used to verify PEs are actually on different nodes."""
        return {
            "rank": self.rank,
            "host": socket.gethostname(),
            "ip": ray.util.get_node_ip_address(),
        }

    def get_unique_id(self):
        import nvshmem.core as nvshmem
        return nvshmem.get_unique_id(empty=False)

    def init(self, uid):

        for k, v in NVSHMEM_ENV.items():
            os.environ.setdefault(k, v)

        import nvshmem.core as nvshmem
        torch.cuda.set_device(0)  

        nvshmem.init(
            uid=uid,
            rank=self.rank,
            nranks=self.world_size,
            initializer_method="uid",
        )
        return {
            "my_pe": nvshmem.my_pe(),
            "n_pes": nvshmem.n_pes(),
        }

    def transport_info(self):
        import nvshmem.core as nvshmem
 
        return {
            "pe": nvshmem.my_pe(),
            "n_pes": nvshmem.n_pes(),
            "requested_transport": os.environ.get("NVSHMEM_REMOTE_TRANSPORT"),
            "ibgda": os.environ.get("NVSHMEM_IB_ENABLE_IBGDA"),
        }

    def run_put(self):
        import nvshmem.core as nvshmem
        stream = torch.cuda.current_stream()

        my_pe = nvshmem.my_pe()
        n_pes = nvshmem.n_pes()

        t = nvshmem.tensor((8,), dtype=torch.float32)
        t.fill_(float(my_pe))
        nvshmem.barrier_all(stream=stream)

        next_pe = (my_pe + 1) % n_pes
        nvshmem.put(t, t, remote_pe=next_pe, stream=stream)
        nvshmem.barrier_all(stream=stream)
        stream.synchronize()

        result = t.clone().cpu().tolist()
        nvshmem.free_tensor(t)
        return {"pe": my_pe, "wrote_to_pe": next_pe, "buffer_now_holds": result}

    def run_get(self):
        import nvshmem.core as nvshmem
        stream = torch.cuda.current_stream()

        my_pe = nvshmem.my_pe()
        n_pes = nvshmem.n_pes()

        src = nvshmem.tensor((8,), dtype=torch.float32)
        src.fill_(float(my_pe))

        dst = nvshmem.tensor((8,), dtype=torch.float32)
        dst.fill_(-1.0)

        nvshmem.barrier_all(stream=stream)

        peer = (my_pe + 1) % n_pes
        nvshmem.get(dst, src, remote_pe=peer, stream=stream)
        nvshmem.barrier_all(stream=stream)
        stream.synchronize()

        result = dst.clone().cpu().tolist()
        nvshmem.free_tensor(src)
        nvshmem.free_tensor(dst)
        return {"pe": my_pe, "read_from_pe": peer, "received": result}

    def run_bandwidth(self, size_bytes: int = 64 * 1024 * 1024, iters: int = 50):
        import time
        import nvshmem.core as nvshmem
        stream = torch.cuda.current_stream()

        my_pe = nvshmem.my_pe()
        n_pes = nvshmem.n_pes()
        peer = (my_pe + 1) % n_pes

        nelems = size_bytes // 4
        t = nvshmem.tensor((nelems,), dtype=torch.float32)
        t.fill_(float(my_pe))
        nvshmem.barrier_all(stream=stream)
        for _ in range(5):
            nvshmem.put(t, t, remote_pe=peer, stream=stream)
        nvshmem.barrier_all(stream=stream)
        stream.synchronize()

        start = time.perf_counter()
        for _ in range(iters):
            nvshmem.put(t, t, remote_pe=peer, stream=stream)
        nvshmem.barrier_all(stream=stream)
        stream.synchronize()
        elapsed = time.perf_counter() - start

        total_bytes = size_bytes * iters
        gbps = total_bytes / elapsed / 1e9

        nvshmem.free_tensor(t)
        return {
            "pe": my_pe,
            "peer": peer,
            "msg_size_MB": size_bytes / 1e6,
            "iters": iters,
            "elapsed_s": round(elapsed, 4),
            "GB_per_s": round(gbps, 2),
        }

    def finalize(self):
        import nvshmem.core as nvshmem
        nvshmem.finalize()


def main():
    ray.init(runtime_env={"env_vars": NVSHMEM_ENV})
    world_size = int(os.environ.get("WORLD_SIZE", "2"))

    workers = [
        RdmaWorker.remote(rank=i, world_size=world_size)
        for i in range(world_size)
    ]

    print("=== placement ===")
    for info in ray.get([w.host_info.remote() for w in workers]):
        print(info)

    uid = ray.get(workers[0].get_unique_id.remote())
    init_results = ray.get([w.init.remote(uid) for w in workers])
    print("\n=== NVSHMEM init ===")
    for r in init_results:
        print(r)

    print("\n===  info ===")
    for r in ray.get([w.transport_info.remote() for w in workers]):
        print(r)

    print("\n=== 1 ===")
    for r in ray.get([w.run_put.remote() for w in workers]):
        print(r)

    print("\n=== 2 ===")
    for r in ray.get([w.run_get.remote() for w in workers]):
        print(r)

    print("\n=== 3 ===")
    for r in ray.get([w.run_bandwidth.remote() for w in workers]):
        print(r)

    ray.get([w.finalize.remote() for w in workers])


if __name__ == "__main__":
    main()
