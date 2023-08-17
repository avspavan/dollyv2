import requests
import json

url = "http://localhost:5000/predict"

story = {
                "story": "summarize the below story to 2 sentences. In a quaint little town, there lived a curious cat named Whiskers. Whiskers had a coat as dark as midnight, adorned with silver streaks that sparkled in the sun. Every morning, he'd venture into the nearby forest, his keen eyes drawn to the mysterious rustling of leaves. One day, following a playful butterfly, he stumbled upon a hidden glen, a haven of wildflowers and shimmering stream. Mesmerized, he returned daily, finding solace in nature's beauty. The townsfolk soon learned of his secret retreat and joined him, creating a heartwarming bond between cat and community. Whiskers' explorations united the town in a shared love for simple joys and hidden wonders."
                }

story_json = json.dumps(story, indent=4)  # Convert dictionary to JSON formatted string with indentation
print(story_json)

headers = {'Content-Type': 'application/json'}  # Set the Content-Type header to indicate JSON data
response = requests.post(url, data=story_json, headers=headers, stream=True)
print(type(response))
for line in response.iter_lines():
            if line:
                            print(f"Received: {line.decode('utf-8')}")

#if response.status_code == 200:
#    response_json = response.json()
#    response_text = response_json["response"]
#    print(response_text)
#else:
#    raise ValueError(f"Failed to predict text: {response.status_code}")
