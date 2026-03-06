import requests

url = "http://134.197.95.82:8000/v1/completions" # 'http://134.197.95.82:8000/v1/completions'


payload = {
    "model": "Qwen/Qwen2-0.5B-Instruct",   # often ignored but still required
    "prompt": "Write a short sentence about GPUs.",
    "max_tokens": 50,
    "temperature": 0
}

response = requests.post(url, json=payload)

print(response.json())
print("\nGenerated text:")
print(response.json()["choices"][0]["text"])