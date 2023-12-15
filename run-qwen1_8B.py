# Load model directly
import transformers

tokenizer = transformers.AutoTokenizer.from_pretrained(
    "Qwen/Qwen-1_8B-Chat", trust_remote_code=True
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-1_8B-Chat", trust_remote_code=True
).eval()

streamer = transformers.TextStreamer(tokenizer)
p = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    streamer=streamer,
    trust_remote_code=True,
)

sequences = p(
    "what is the recipe of mayonnaise?",
    temperature=0.9,
    top_k=50,
    top_p=0.9,
    max_length=500,
)
