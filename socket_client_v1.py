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
    prompt = "Summarize the paragraph. Linux is a versatile and open-source operating system renowned for its stability, security, and flexibility. Developed as a Unix-like system, Linux has gained widespread popularity due to its ability to run on a wide range of hardware, from servers and desktop computers to embedded devices and smartphones. Its modular design allows users to choose from various distributions, each tailored to specific needs and preferences. Linux's robust command-line interface empowers users with fine-grained control over system configurations and tasks, making it a favorite among developers, system administrators, and tech enthusiasts. The collaborative nature of the open-source community has led to continuous innovation and rapid evolution, ensuring that Linux remains at the forefront of modern computing."
    generated_text = get_generated_text(prompt)