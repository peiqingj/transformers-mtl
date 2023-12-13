# Load model directly
import transformers


streamer=transformers.TextStreamer(None)
p = transformers.pipeline(
    'text-generation',
    model='meta-llama/Llama-2-7b-chat-hf',
    tokenizer='meta-llama/Llama-2-7b-chat-hf',
    streamer=streamer
)
streamer.tokenizer = p.tokenizer
sequences = p(
    'what is the recipe of mayonnaise?',
    temperature=0.9,
    top_k=50,
    top_p=0.9,
    max_length=500,
)
