import ray
import cupy as cp
import ray.util.collective as collective


@ray.remote(num_gpus=1)
class Worker:
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        # GPU buffer filled with (rank + 1)
        self.buf = cp.ones(4, dtype=cp.float32) * (rank + 1)

    def setup(self, group_name="default"):
        collective.init_collective_group(
            world_size=self.world_size,
            rank=self.rank,
            backend="nccl",
            group_name=group_name,
        )
        return True

    def allreduce(self, group_name="default"):
        before = self.buf.get().tolist()
        collective.allreduce(self.buf, group_name=group_name)
        cp.cuda.Stream.null.synchronize()
        return before, self.buf.get().tolist()

    def broadcast_from_rank0(self, group_name="default"):
        if self.rank == 0:
            self.buf = cp.arange(4, dtype=cp.float32) * 10  # [0, 10, 20, 30]
        collective.broadcast(self.buf, src_rank=0, group_name=group_name)
        cp.cuda.Stream.null.synchronize()
        return self.buf.get().tolist()

    def destroy(self, group_name="default"):
        collective.destroy_collective_group(group_name)


if __name__ == "__main__":
    ray.init()

    world_size = 2
    workers = [Worker.remote(rank=i, world_size=world_size)
               for i in range(world_size)]

    # All ranks must call init concurrently — hence parallel ray.get
    ray.get([w.setup.remote() for w in workers])

    print("AllReduce (sum):")
    for rank, (before, after) in enumerate(
        ray.get([w.allreduce.remote() for w in workers])
    ):
        print(f"  Rank {rank}: before={before} after={after}")

    print("\nBroadcast from rank 0:")
    for rank, result in enumerate(
        ray.get([w.broadcast_from_rank0.remote() for w in workers])
    ):
        print(f"  Rank {rank}: {result}")

    ray.get([w.destroy.remote() for w in workers])
    ray.shutdown()