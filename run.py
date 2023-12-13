# Load model directly
import transformers
from modeling_llama import LlamaAttentionNPU

# hacking eager LlamaAttention by LlamaAttentionNPU
mode = 'eager'
transformers.models.llama.modeling_llama.LLAMA_ATTENTION_CLASSES[mode] = LlamaAttentionNPU  
streamer=transformers.TextStreamer(None)
p = transformers.pipeline(
    'text-generation',
    model=transformers.LlamaForCausalLM.from_pretrained('meta-llama/Llama-2-7b-chat-hf', attn_implementation=mode),
    tokenizer='meta-llama/Llama-2-7b-chat-hf',
    streamer=streamer,
)
streamer.tokenizer = p.tokenizer

# verify the hacking is loaded
assert id(p.model.model.layers[0].self_attn.__class__) == id(LlamaAttentionNPU), 'hacking failed'

sequences = p(
    'what is the recipe of mayonnaise?',
    temperature=0.9,
    top_k=50,
    top_p=0.9,
    max_length=500,
)
