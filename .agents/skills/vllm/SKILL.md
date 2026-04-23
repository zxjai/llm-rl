# vLLM Weight Transfer Mechanism for RL


Recall the typical RL process

1. Generate rollouts (vLLM inference)
2. Puase vLLM when new weights are available
3. Sync weights
4. Resume generation

vLLM has an engine called AsyncLLM used for model inference (step 1), it provides two methods to handle steps 2 and 4.

```python
# vllm/v1/engine/async_llm.py

class AsyncLLM(EngineClient):
    async def pause_generation(...):
        ...
    async def resume_generation(self):
        ...
```

In order to transfer weights (step 2) first establish a communication channel. The vLLM server has the following Python API (although HTTP endpoints are available we shall not consider those).

```python
# vllm/v1/engine/async_llm.py

class AsyncLLM(EngineClient):
    async def init_weight_transfer_engine(...): # takes WeightTransferInitRequest (IP:port etc)
        ...
    async def update_weights(...):
        ...
```

Use the `WeightTransferConfig` to select NCCL communication backend to transfer between GPUS, otherwise use IPC backend if new and old weights are co-located on the same GPU:

```python
from vllm import LLM
from vllm.config import WeightTransferConfig

llm = LLM(
    model="my-model",
    weight_transfer_config=WeightTransferConfig(backend="nccl"),  # or 'ipc' for cuda interprocess communication
)
```

Trainer side APIs are 

```python
# handshake
NCCLWeightTransferEngine.trainer_init(init_info) # (IP:port etc)

# send weights
NCCLWeightTransferEngine.trainer_send_weights(
    iterator=model.named_parameters(),
    trainer_args=backend_specific_args, # NCCLTrainerSendWeightsArgs
)
```

## Advanced 

The weight transfer mechanism can be customized (for example to use RDMA backend) by inheriting from `WeightTransferEngine` ABC [base class](https://docs.vllm.ai/en/stable/api/vllm/distributed/weight_transfer/base/), which defines a contract between vLLM worker and the transport backend, and registering the new class with [the factory class](https://docs.vllm.ai/en/stable/api/vllm/distributed/weight_transfer/factory/#vllm.distributed.weight_transfer.factory.WeightTransferEngineFactory) `WeightTransferEngineFactory`.

## Examples

Trainer calls `NCCLWeightTransferEngine.trainer_init()` for address and port, which returns a group that `NCCLWeightTransferEngine.trainer_send_weights()` takes alongside the weights for communication.

The inference engine initializes with a weight transfer backend 

```py
llm = LLM(
    weight_transfer_config=WeightTransferConfig(backend="nccl")
)
```

then gets to know communication address and port with `llm.init_weight_transfer_engine()`. To recieve weights `llm.update_weights()`.

The basic architecture is that sender and receiver establish communication. Then sender emits message, receiver gets message.