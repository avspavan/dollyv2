import requests
import json

def send_text_to_server(text):
    url = 'http://localhost:5000/predict'
    data = {'text': text}
    headers = {'Content-Type': 'application/json'}  # Set the Content-Type header to indicate JSON data
    data_json = json.dumps(data)  # Convert dictionary to JSON-formatted string

    response = requests.post(url, data=data_json, headers=headers)  # Send JSON-formatted string as data
    return response.json()

if __name__ == '__main__':
    input_text = "my my name name is is the the dude dude"
    response_data = send_text_to_server(input_text)

    repeated_text = response_data['repeated_text']
    print("Modified text from server:")
    print(repeated_text)
