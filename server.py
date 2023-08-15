import flask
import json
import transformers

app = flask.Flask(__name__)

#model = transformers.AutoModelForSeq2SeqLM.from_pretrained("roberta-base")

@app.route("/predict", methods=["POST"])
def predict():
    """
    This function takes a text input as a POST request and returns the predicted text from the LLM model.
    """
    print("************************************************")
    print("this is from the server")
    print("************************************************")
    data = json.loads(flask.request.data)
    text_file = data["text"]
    print("this is the server output", text_file)
    response = text_file
    return json.dumps({"response": response})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
