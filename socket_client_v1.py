import socket

def get_generated_text(prompt):
    host = "localhost"
    port = 8080

    # Create a socket object
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Connect to the server
    sock.connect((host, port))

    # Send the prompt to the server
    sock.sendall(prompt.encode())

    # Receive the generated text from the server
    generated_text = ""
    while True:
        data = sock.recv(1024).decode()
        print(data, end="", flush=True)
        if data == "STREAMING_COMPLETE":
            break
        generated_text += data

    # Close the connection
    sock.close()

    return generated_text

if __name__ == "__main__":
    file_path = "./virat.txt"  
    try:
        with open(file_path, 'r') as file:
            file_content = file.read()
            print("File content:\n", file_content)  # Print the content of the file
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except Exception as e:
        print("An error occurred:", str(e))

    prompt = "Summarize the paragraph" + file_content
    generated_text = get_generated_text(prompt)