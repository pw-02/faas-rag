import openai

client = openai.OpenAI(
    base_url="http://134.197.95.82:8000/v1",
    api_key="EMPTY",
)

resp = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=[
        {"role": "user", "content": "Say hello in one word."}
    ],
    temperature=0.0,
    max_tokens=10,
)

print("Response:")
print(resp.choices[0].message.content)

print("\nUsage:")
print(resp.usage)
