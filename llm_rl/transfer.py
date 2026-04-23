import asyncio
import uuid
from dataclasses import asdict
from pathlib import Path

import ray
import torch
import vllm
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import SamplingParams
from vllm.config import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (WeightTransferInitRequest,
                                                   WeightTransferUpdateRequest)
from vllm.distributed.weight_transfer.nccl_engine import (
    NCCLTrainerSendWeightsArgs, NCCLWeightTransferEngine,
    NCCLWeightTransferInitInfo, NCCLWeightTransferUpdateInfo)
from vllm.utils.network_utils import get_ip, get_open_port
from vllm.v1.executor import Executor


class RolloutWorker(vllm.AsyncLLMEngine):
    def __init__(self, max_tokens=10, **kwargs):
        engine_args = vllm.AsyncEngineArgs(**kwargs)
        vllm_config = engine_args.create_engine_config()
        executor_class = Executor.get_class(vllm_config)
        super().__init__(
            vllm_config=vllm_config, 
            executor_class=executor_class,
            log_requests=True,
            log_stats=True
        )
        self.max_tokens = max_tokens

        self._request_pause_flag = False
        self._generation_paused = False

    async def do_generate(
        self, prompt_token_ids: list[int], sampling_params: vllm.SamplingParams
    ) -> tuple[vllm.RequestOutput, int]:
        """
        Single rollout. Returns output and an index specifying number of tokens
        generated prior to weight update if weighted changed, else -1.
        """ 
        pause_token_index = -1
        previous_token_count = 0
        async for request_output in self.generate(
            {'prompt_token_ids': prompt_token_ids},
            sampling_params,
            request_id=str(uuid.uuid4())
        ):
            output = request_output
            current_token_count = len(output.outputs[0].token_ids)
            if (
                current_token_count >= self.max_tokens
                and not self._request_pause_flag
            ):
                self._request_pause_flag = True
            if self._generation_paused and pause_token_index == -1:
                pause_token_index = previous_token_count
            previous_token_count = current_token_count
        return output, pause_token_index

    async def pause_after_n_tokens(self):
        """
        Wait for a request to set pause flag, then pause.
        """
        while not self._request_pause_flag:
            await asyncio.sleep(0)
        await super().pause_generation(mode='keep')
        await asyncio.sleep(5)
        self._generation_paused = True

@ray.remote(num_gpus=1)
class Trainer:
    def __init__(self, model_path: str):
        from vllm.model_executor.layers.batch_invariant import \
            init_batch_invariance
        from vllm.v1.attention.backends.registry import AttentionBackendEnum
        init_batch_invariance(AttentionBackendEnum.FLASH_ATTN)

        # model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16
        ).to('cuda:0')

        self.port = get_open_port()
        self.master_address = get_ip()

    def get_master_address_and_port(self):
        return self.master_address, self.port

    def get_weight_metadata(self):
        """
        Metadata conssits of weight name, dtype, shapes
        """
        names = []
        dtype_names = []
        shapes = []
        for name, p in self.model.named_parameters():
            names.append(name)
            dtype_names.append(str(p.dtype).split('.')[-1])
            shapes.append(list(p.shape))
        return names, dtype_names, shapes

    def init_weight_transfer_group(self, world_size: int):
        self.model_update_group = NCCLWeightTransferEngine.trainer_init(
            dict(
                master_address=self.master_address,
                master_port=self.port,
                world_size=world_size
            )
        )
    
    def broadcast_weights(self, packed: bool = True):
        trainer_args = NCCLTrainerSendWeightsArgs(
            group=self.model_update_group,
            packed=packed
        )

        NCCLWeightTransferEngine.trainer_send_weights(
            iterator=self.model.named_parameters(),
            trainer_args=trainer_args
        )

    @torch.inference_mode()
    def generate(self, token_ids: list[int], max_new_tokens: int) -> list[int]:
        input_ids = torch.tensor([token_ids], device='cuda:0')
        output = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )
        new_token_ids = output[0, len(token_ids) :].tolist()
        return new_token_ids

ray_env_vars = {
    'RAY_EXPERIMENTAL_NOSET_CUDA_ENV_VAR': '1',
    'VLLM_BATCH_INVARIANT': '1'
}

ray.init(runtime_env={'env_vars': ray_env_vars})

trainer = Trainer.remote('models/gptoss_20b')

llm_kwargs = dict(
    model='models/gptoss_20b',
    enforce_eager=True,
    max_model_len=8192,
    gpu_memory_utilization=0.75,
    distributed_executor_backend='ray',
    attention_backend='FLASH_ATTN',
    weight_transfer_config=WeightTransferConfig(backend='nccl')
)

llm = ray.remote(num_cpus=0, num_gpus=0)(RolloutWorker).remote(**llm_kwargs)

PROMPTS = [
    "The capital of France is",
    "The continent of the United States is",
    "The largest ocean on Earth is",
]

tokenizer = AutoTokenizer.from_pretrained('models/gptoss_20b')
batch_prompt_token_ids = [
    tokenizer.encode(prompt, add_special_tokens=False) for prompt in PROMPTS
]

master_address, master_port = ray.get(trainer.get_master_address_and_port.remote())
world_size = 2
inference_handle = llm.init_weight_transfer_engine.remote(
    WeightTransferInitRequest(
        init_info=asdict(
            NCCLWeightTransferInitInfo(
                master_address=master_address,
                master_port=master_port,
                rank_offset=1,
                world_size=world_size
            )
        )
    )
)
train_handle = trainer.init_weight_transfer_group.remote(world_size)
ray.get([train_handle, inference_handle])


names, dtype_names, shapes = ray.get(trainer.get_weight_metadata.remote())

print(f"\n{'=' * 50}")
print(f"Prompts ({len(PROMPTS)}):")
for p in PROMPTS:
    print(f"  - {p!r}")
print(f"{'=' * 50}")

sampling_params = SamplingParams(
    temperature=0, max_tokens=110
)

gen_futures = [
    llm.do_generate.remote(ptids, sampling_params) for ptids in batch_prompt_token_ids
]

ray.get(llm.pause_after_n_tokens.remote())

inference_handle = llm.update_weights.remote(
    WeightTransferUpdateRequest(
        update_info=asdict(
            NCCLWeightTransferUpdateInfo(
                names=names,
                dtype_names=dtype_names,
                shapes=shapes,
                packed=True,
            )
        )
    )
)
train_handle = trainer.broadcast_weights.remote(packed=True)
ray.get([train_handle, inference_handle])

ray.get(llm.resume_generation.remote())
results = ray.get(gen_futures)

for i, (output, pause_idx) in enumerate(results):
    all_token_ids = list(output.outputs[0].token_ids)
    before_text = tokenizer.decode(all_token_ids[:pause_idx])
    after_text = tokenizer.decode(all_token_ids[pause_idx:])
    print(f"\n  Request {i} ({PROMPTS[i]!r}):")
    print(f"    Old weights ({pause_idx} tokens): {before_text!r}")
    n_after = len(all_token_ids) - pause_idx
    print(f"    New weights ({n_after} tokens): {after_text!r}")


ray.get(llm.shutdown.remote())
ray.kill(llm)
ray.kill(trainer)
