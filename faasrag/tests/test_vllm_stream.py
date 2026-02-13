import openai
import time

client = openai.OpenAI(
    base_url="http://134.197.95.82:8000/v1",
    api_key="EMPTY",
)

start_time = time.time()
first_token_time = None

stream = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Say hello in one word."}],
    stream=True,
    max_tokens=10,
    temperature=0.0,
    stream_options={"include_usage": True},
)

full_text = ""
usage = None

for chunk in stream:
    if chunk.choices:
        delta = chunk.choices[0].delta.content
        if delta:
            if first_token_time is None:
                first_token_time = time.time()
            full_text += delta

    if getattr(chunk, "usage", None) is not None:
        usage = chunk.usage

end_time = time.time()

ttft_s = (first_token_time - start_time) if first_token_time else None
total_s = end_time - start_time

# Derived rates
prefill_tps = None
decode_tps = None
if usage and ttft_s and ttft_s > 0:
    prefill_tps = usage.prompt_tokens / ttft_s

if usage and ttft_s is not None:
    decode_time_s = max(total_s - ttft_s, 1e-9)
    decode_tps = usage.completion_tokens / decode_time_s

print("Response:", full_text)
print("Usage:", usage)
print("TTFT (s):", ttft_s)
print("Total time (s):", total_s)
print("Prefill tokens/sec:", prefill_tps)
print("Decode tokens/sec:", decode_tps)
