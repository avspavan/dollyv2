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
import traceback
import os

# Get the model name from the environment variable or use the default
#models included in this image: dolly-v2-3b, llama7b
# choose the model in the dockerfile or here
model_name = os.environ.get('DEFAULT_MODEL', 'llama7b')

#model_id = "databricks/dolly-v2-3b"
#model_id = "meta-llama/Llama-2-13b-chat"
#model_path = Path("dolly-v2-3b")
#model_path = Path("/home/ubuntu/pavan/openvino_notebooks/notebooks/240-dolly-2-instruction-following/llama7b")
core = Core()
device = 'CPU'

model_path = Path(model_name)

#tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_path)

current_device = device

if model_path.exists():
    ov_model = OVModelForCausalLM.from_pretrained(model_path, device=current_device)
else:
    print("couldn't find the model")
    exit
    ov_model = OVModelForCausalLM.from_pretrained(model_id, device=current_device, from_transformers=True)
    ov_model.save_pretrained(model_path)

DEFAULT_SYSTEM_PROMPT = """\

You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\

"""

INSTRUCTION_KEY = "### Instruction:"
RESPONSE_KEY = "### Response:"
END_KEY = "### End"
INTRO_BLURB = (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request." + DEFAULT_SYSTEM_PROMPT
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


def generate_response(conn):
    print(" ")
    print("inside generate_response")
    while True:
        data = conn.recv(2048)
        #print("data", data)
        if data == b"":
           print(" ")
           print("Client disconnected")
           conn.close()
           break
        # Get the prompt from the client
        prompt = data.decode()
        print("This is the input from the client: ", prompt)

        # Prepare input prompt according to model expected template
        prompt_text = PROMPT_FOR_GENERATION_FORMAT.format(instruction=prompt)
        print(" ")
        print("___________________________________________________________________________________")
        print(" ")
        print("This is the entire prompt: ", prompt_text)
        print(" ")
        print("___________________________________________________________________________________")
        print(" ")

        # Tokenize the user text.
        model_inputs = tokenizer(prompt_text, return_tensors="pt")

        # Start generation on a separate thread, so that we don't block the UI. The text is pulled from the streamer
        # in the main thread. Adds timeout to the streamer to handle exceptions in the generation thread.
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=10)
        generate_kwargs = dict(
            model_inputs,
            streamer=streamer,
            max_new_tokens=1024,
            do_sample=True,
            top_p=0.92,
            temperature=0.8,
            top_k=0,
            eos_token_id=end_key_token_id
        )
        t = Thread(target=ov_model.generate, kwargs=generate_kwargs)
        try:
            t.start()
        except RuntimeError as e:
            if "REQUEST_BUSY" in str(e):
                print("Inference pipeline is busy, waiting...")
                time.sleep(10)
                t.start()
        print("thread started to get response")

        # Pull the generated text from the streamer, and update the model output.
        model_output = ""
        per_token_time = []
        num_tokens = 0
        try:
            for new_text in streamer:
                conn.sendall(new_text.encode())
                print(new_text, end="")    
                sys.stdout.flush()
                model_output += new_text
        except RuntimeError as e:
            if "REQUEST_BUSY" in str(e):
                print("Inference pipeline is busy, waiting...")
                conn.sendall("Inference pipeline is busy, please wait.".encode())
                time.sleep(2)  # Adjust the sleep interval as needed
                continue  # Retry the inference
            else:
                print("Exception during inference:", e)
                conn.sendall("An error occurred during inference.".encode())
                break
        except ConnectionResetError:
            print("Client disconnected")
            conn.close()
            break
        finally:
            if conn:
                try:
                    done_message = "STREAMING_COMPLETE".encode()
                    conn.sendall(done_message, 2048)
                    time.sleep(5)
                    # Check if the client is still connected
                    data = conn.recv(1024)
                    #print("data", data)
                    if data == b"":
                       print(" ")
                       print("Client disconnected")
                    conn.close()
                    break
                except Exception as e:
                    print("Error while closing connection:", e)
            print("Connection closed")
    return

def handle_client(conn):
    try:
        generate_response(conn)
    except Exception as e:
        print("Exception in thread:", e)
        time.sleep(5)
    finally:
        conn.close()

def run_server():
    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            host = "0.0.0.0"
            port = 8000
            sock.bind((host, port))
            sock.listen(5)
            
            while True:
                try:
                    conn, addr = sock.accept()
                    print("connected to:", addr)
                    t = Thread(target=handle_client, args=(conn,))
                    t.start()
                except KeyboardInterrupt:
                    print("Server shutting down...")
                    break
                except Exception as e:
                    print("Exception in main loop:", e)
        except Exception as e:
            print("Server crashed:", e)
            traceback.print_exc()
        finally:
            sock.close()
            print("Restarting server in 5 seconds...")
            time.sleep(5)


if __name__ == "__main__":
    try:
        run_server()
    except Exception:
        print("throwing exception from run_server() code")
        time.sleep(5)
        run_server()
