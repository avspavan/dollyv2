from flask import Flask, request, Response
import time

app = Flask(__name__)

def generate_response(query):
    # Replace this with your actual response generation logic
    response = f"Server response to: {query}\n"
    return response

@app.route('/process_query', methods=['POST'])
def process_query():
    query = request.json.get('query', '')
    response_stream = generate_response(query)

    def generate():
        while True:
            yield response_stream
            time.sleep(1)

    return Response(generate(), mimetype='text/plain')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
