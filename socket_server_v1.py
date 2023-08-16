from threading import Thread
from transformers import AutoTokenizer, TextIteratorStreamer
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
from optimum.intel.openvino import OVModelForCausalLM
from openvino.runtime import Core
import socket
import time
import sys

core = Core()
device = 'CPU'
model_id = "databricks/dolly-v2-3b"
model_path = Path("dolly-v2-3b")

tokenizer = AutoTokenizer.from_pretrained(model_id)

current_device = device

if model_path.exists():
    ov_model = OVModelForCausalLM.from_pretrained(model_path, device=current_device)
else:
    ov_model = OVModelForCausalLM.from_pretrained(model_id, device=current_device, from_transformers=True)
    ov_model.save_pretrained(model_path)

INSTRUCTION_KEY = "### Instruction:"
RESPONSE_KEY = "### Response:"
END_KEY = "### End"
INTRO_BLURB = (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request."
)

# This is the prompt that is used for generating responses using an already trained model.  It ends with the response
# key, where the job of the model is to provide the completion that follows it (i.e. the response itself).
PROMPT_FOR_GENERATION_FORMAT = """{intro}

{instruction_key}
{instruction}

{response_key}
""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    response_key=RESPONSE_KEY,
)


def get_special_token_id(tokenizer: AutoTokenizer, key: str) -> int:
    """
    Gets the token ID for a given string that has been added to the tokenizer as a special token.

    When training, we configure the tokenizer so that the sequences like "### Instruction:" and "### End" are
    treated specially and converted to a single, new token.  This retrieves the token ID each of these keys map to.

    Args:
        tokenizer (PreTrainedTokenizer): the tokenizer
        key (str): the key to convert to a single token

    Raises:
        RuntimeError: if more than one ID was generated

    Returns:
        int: the token ID for the given key
    """
    token_ids = tokenizer.encode(key)
    if len(token_ids) > 1:
        raise ValueError(f"Expected only a single token for '{key}' but found {token_ids}")
    return token_ids[0]

tokenizer_response_key = next((token for token in tokenizer.additional_special_tokens if token.startswith(RESPONSE_KEY)), None)

end_key_token_id = None
if tokenizer_response_key:
    try:
        end_key_token_id = get_special_token_id(tokenizer, END_KEY)
        print("end key token id", end_key_token_id)
        # Ensure generation stops once it generates "### End"
    except ValueError:
        pass

def run_server():
    # Create a socket object
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Bind the socket to a port
    host = "localhost"
    port = 8080
    sock.bind((host, port))

    # Listen for connections
    sock.listen(5)

    while True:
        # Accept a connection
        conn, addr = sock.accept()
        print("connected to:", addr)

        def generate_response():
            print("inside generate_response")
            while conn.connect:
                data = conn.recv(1024)
                if data == b"":
                   print("Client disconnected")
                   conn.close()
                   break
                # Get the prompt from the client
                prompt = conn.recv(1024).decode()
                #print("this is the prompt", prompt)

                # Prepare input prompt according to model expected template
                prompt_text = PROMPT_FOR_GENERATION_FORMAT.format(instruction=prompt)
                print("this is the prompt", prompt_text)

                # Tokenize the user text.
                model_inputs = tokenizer(prompt_text, return_tensors="pt")

                # Start generation on a separate thread, so that we don't block the UI. The text is pulled from the streamer
                # in the main thread. Adds timeout to the streamer to handle exceptions in the generation thread.
                streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
                generate_kwargs = dict(
                    model_inputs,
                    streamer=streamer,
                    max_new_tokens=512,
                    do_sample=True,
                    top_p=0.92,
                    temperature=0.8,
                    top_k=0,
                    eos_token_id=end_key_token_id
                )
                t = Thread(target=ov_model.generate, kwargs=generate_kwargs)
                t.start()
                print("thread started to get response")

                # Pull the generated text from the streamer, and update the model output.
                model_output = ""
                per_token_time = []
                num_tokens = 0
                for new_text in streamer:
                    print(new_text, end="")    
                    sys.stdout.flush()
                    conn.sendall(new_text.encode())
                    model_output += new_text
                # Send a message to the client to let it know that the streaming is done
                done_message = "STREAMING_COMPLETE".encode()
                conn.sendall(done_message)
                time.sleep(5)
                # Check if the client is still connected
                data = conn.recv(1024)
                if data == b"":
                   print("Client disconnected")
                   conn.close()
                   break

            # Close the connection
            conn.close()

        generate_response()

    return

if __name__ == "__main__":
    try:
        run_server()
    except Exception:
        print("throwing exception")