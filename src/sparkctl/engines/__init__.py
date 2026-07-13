"""Engine registry: how a service's model server is launched/torn down."""
from sparkctl.engines import ollama, vllm

ENGINES = {"vllm": (vllm.vllm_up, vllm.vllm_down), "ollama": (ollama.ollama_up, ollama.ollama_down)}
