import requests

text = open('text.txt', 'r', encoding="utf8").read()

query = {
    "text": text,
    "temperature": 2,
    "max_tokens": 2000,
    "top_p": 0.2
}
url = "http://localhost:8000/summarize"

response = requests.post(url, headers={'Content-Type': 'application/json'}, json=query, stream=True)

for chunk in response.iter_content(chunk_size=10000):
    if chunk:
        print(chunk.decode("utf-8")[5:])  # parsing the data: prefix
