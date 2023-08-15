import requests
import threading

def receive_response(response):
    for chunk in response.iter_content(chunk_size=1024):
        if chunk:
            print(chunk.decode(), end='')

def start_client():
    url = 'http://localhost:5000/process_query'  # Replace 'server_ip' with the server's IP
    
    while True:
        query = input("Enter your query (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break

        response = requests.post(url, json={'query': query}, stream=True)

        if response.status_code == 200:
            response_thread = threading.Thread(target=receive_response, args=(response,))
            response_thread.start()
        else:
            print("Error: Unable to connect to the server.")

if __name__ == "__main__":
    start_client()
