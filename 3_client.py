import requests
import json


def validate_json(json_data):
    try:
        json.loads(json_data)
        return True
    except json.JSONDecodeError:
        return False

with open("poem.txt", "r") as f:
    text = f.read()


url = "http://localhost:5000/predict"
data = {"text": text}
headers = {"Content-Type": "application/json"}
print("data", data["text"])
print(validate_json(data))
json_data = 'This is not a valid JSON object'
print(validate_json(json_data))

response = requests.post(url, data=data, headers=headers)

if response.status_code == 200:
    response_json = response.json()
    response_text = response_json["response"]
    print(response_text)
else:
    raise ValueError(f"Failed to predict text: {response.status_code}")
