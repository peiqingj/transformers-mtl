# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LLAMA_ATTENTION_CLASSES

from modeling_llama import LlamaAttentionNPU
LLAMA_ATTENTION_CLASSES["LlamaAttentionNPU"] = LlamaAttentionNPU
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
print(model.name)
