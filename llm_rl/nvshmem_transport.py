import os
import ray
import torch


NVSHMEM_ENV = {
    # Single-node: use P2P only, skip IB entirely
    "NVSHMEM_REMOTE_TRANSPORT": "none",
    "NVSHMEM_IB_ENABLE_IBGDA": "0",
    "NVSHMEM_DISABLE_P2P": "0",
    # Disable the proxy thread (needed only for IB / GPU-initiated remote ops)
    "NVSHMEM_DISABLE_NCCL": "1",
    # Heap size
    "NVSHMEM_SYMMETRIC_SIZE": "1G",
    # Bootstrap
    "NVSHMEM_BOOTSTRAP": "UID",
    # Optional: verbose logs if it still fails
    # "NVSHMEM_DEBUG": "INFO",
    # "NVSHMEM_INFO": "1",
}


@ray.remote(num_gpus=1)
class NvshmemWorker:
    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size

    def get_unique_id(self):
        import nvshmem.core as nvshmem
        self.uid = nvshmem.get_unique_id(empty=False)
        return self.uid

    def init(self, uid):
        # IMPORTANT: set env vars BEFORE importing nvshmem in this process
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
        return nvshmem.my_pe(), nvshmem.n_pes()

    def run(self):
        import nvshmem.core as nvshmem

        stream = torch.cuda.current_stream()
        my_pe = nvshmem.my_pe()
        n_pes = nvshmem.n_pes()

        # Symmetric tensor — one per PE
        t = nvshmem.tensor((8,), dtype=torch.float32)
        t.fill_(float(my_pe))

        nvshmem.barrier_all(stream=stream)

        # Put my buffer into the SAME symmetric buffer on the next PE
        next_pe = (my_pe + 1) % n_pes
        nvshmem.put(t, t, remote_pe=next_pe, stream=stream)

        nvshmem.barrier_all(stream=stream)
        stream.synchronize()

        result = t.clone().cpu().tolist()
        nvshmem.free_tensor(t)
        return {"pe": my_pe, "received": result}

    def finalize(self):
        import nvshmem.core as nvshmem
        nvshmem.finalize()


def main():
    # Forward NVSHMEM env to every Ray worker
    ray.init(runtime_env={"env_vars": NVSHMEM_ENV})

    world_size = 2
    workers = [
        NvshmemWorker.remote(rank=i, world_size=world_size)
        for i in range(world_size)
    ]

    uid = ray.get(workers[0].get_unique_id.remote())
    print(ray.get([w.init.remote(uid) for w in workers]))

    for r in ray.get([w.run.remote() for w in workers]):
        print(r)

    ray.get([w.finalize.remote() for w in workers])


if __name__ == "__main__":
    main()