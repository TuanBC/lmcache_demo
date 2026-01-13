import requests

headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer vllm_sk_b6c40e272ffc3dabf0116bcc743d288d4e938776061606dccf070b15",
}

json_data = {
    "model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "messages": [
        {
            "role": "user",
            "content": "Hello!",
        },
    ],
    "max_tokens": 100,
    "temperature": 0.7,
}

response = requests.post(
    "http://89.169.108.198:30080/v1/chat/completions", headers=headers, json=json_data
)
print(response.json())
