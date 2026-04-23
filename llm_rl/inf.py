# from vllm import LLM

# import os
# os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
# print('#'*50 + 'vLLM'+'#'*50)

# llm = LLM(
#     model="models/gptoss_20b",
#     enforce_eager=True,
#     max_model_len=8192,
#     gpu_memory_utilization=0.75,
#     quantization=None,
#     # no distributed_executor_backend — defaults to single-process
# )

# def _collect(self, filter_substr=None):
#     # `self` is the Worker inside the engine process
#     model = self.model_runner.model
#     rows = []
#     for n, p in model.named_parameters():
#         if filter_substr and filter_substr not in n:
#             continue
#         rows.append((n, tuple(p.shape), str(p.dtype)))
#     for n, b in model.named_buffers():
#         if filter_substr and filter_substr not in n:
#             continue
#         rows.append((n + "  [buffer]", tuple(b.shape), str(b.dtype)))
#     return rows

# results = llm.collective_rpc(_collect, args=("layers.0",))
# for row in results[0]:
#     print(row)

import os, pickle
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
from vllm import LLM
llm = LLM(model="models/gptoss_20b", enforce_eager=True,
            max_model_len=8192, gpu_memory_utilization=0.75)
def collect(worker):
    model = worker.model_runner.model
    params = [(n, list(p.shape), str(p.dtype).split(".")[-1])
                for n, p in model.named_parameters()]
    buffers = [(n, list(b.shape), str(b.dtype).split(".")[-1])
                for n, b in model.named_buffers()]
    return {"params": params, "buffers": buffers}
spec = llm.collective_rpc(collect)[0]
pickle.dump(spec, open("vllm_full_spec.pkl", "wb"))