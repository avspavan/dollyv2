from threading import Thread
from time import perf_counter
from typing import List
import gradio as gr
from transformers import AutoTokenizer, TextIteratorStreamer
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
from optimum.intel.openvino import OVModelForCausalLM
#import ipywidgets as widgets
from openvino.runtime import Core
import socket
import time

core = Core()

#device = widgets.Dropdown(
#    options=core.available_devices + ["AUTO"],
#    value='AUTO',
#    description='Device:',
#    disabled=False,
#)

device = 'CPU'


model_id = "databricks/dolly-v2-3b"
model_path = Path("dolly-v2-3b")

tokenizer = AutoTokenizer.from_pretrained(model_id)

current_device = device
#current_device = device.value

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
        # Ensure generation stops once it generates "### End"
    except ValueError:
        pass

def run_generation(user_text:str, top_p:float, temperature:float, top_k:int, max_new_tokens:int, perf_text:str):
    """
    Text generation function
    
    Parameters:
      user_text (str): User-provided instruction for a generation.
      top_p (float):  Nucleus sampling. If set to < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for a generation.
      temperature (float): The value used to module the logits distribution.
      top_k (int): The number of highest probability vocabulary tokens to keep for top-k-filtering.
      max_new_tokens (int): Maximum length of generated sequence.
      perf_text (str): Content of text field for printing performance results.
    Returns:
      model_output (str) - model-generated text
      perf_text (str) - updated perf text filed content
    """
    
    # Prepare input prompt according to model expected template
    print(user_text)
    prompt_text = PROMPT_FOR_GENERATION_FORMAT.format(instruction=user_text)
    
    # Tokenize the user text.
    model_inputs = tokenizer(prompt_text, return_tensors="pt")

    # Start generation on a separate thread, so that we don't block the UI. The text is pulled from the streamer
    # in the main thread. Adds timeout to the streamer to handle exceptions in the generation thread.
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        model_inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=top_p,
        temperature=float(temperature),
        top_k=top_k,
        eos_token_id=end_key_token_id
    )
    t = Thread(target=ov_model.generate, kwargs=generate_kwargs)
    t.start()

    # Pull the generated text from the streamer, and update the model output.
    model_output = ""
    per_token_time = []
    num_tokens = 0
    start = perf_counter()
    for new_text in streamer:
        current_time = perf_counter() - start
        model_output += new_text
        perf_text, num_tokens = estimate_latency(current_time, perf_text, new_text, per_token_time, num_tokens)
        yield model_output, perf_text
        start = perf_counter()
    return model_output, perf_text

def estimate_latency(current_time:float, current_perf_text:str, new_gen_text:str, per_token_time:List[float], num_tokens:int):
    """
    Helper function for performance estimation

    Parameters:
      current_time (float): This step time in seconds.
      current_perf_text (str): Current content of performance UI field.
      new_gen_text (str): New generated text.
      per_token_time (List[float]): history of performance from previous steps.
      num_tokens (int): Total number of generated tokens.

    Returns:
      update for performance text field
      update for a total number of tokens
    """
    num_current_toks = len(tokenizer.encode(new_gen_text))
    num_tokens += num_current_toks
    per_token_time.append(num_current_toks / current_time)
    if len(per_token_time) > 10 and len(per_token_time) % 4 == 0:
        current_bucket = per_token_time[:-10]
        return f"Average generation speed: {np.mean(current_bucket):.2f} tokens/s. Total generated tokens: {num_tokens}", num_tokens
    return current_perf_text, num_tokens

def reset_textbox(instruction:str, response:str, perf:str):
    """
    Helper function for resetting content of all text fields

    Parameters:
      instruction (str): Content of user instruction field.
      response (str): Content of model response field.
      perf (str): Content of performance info filed

    Returns:
      empty string for each placeholder
    """
    return "", "", ""


def select_device(device_str:str, current_text:str = "", progress:gr.Progress = gr.Progress()):
    """
    Helper function for uploading model on the device.

    Parameters:
      device_str (str): Device name.
      current_text (str): Current content of user instruction field (used only for backup purposes, temporally replacing it on the progress bar during model loading).
      progress (gr.Progress): gradio progress tracker
    Returns:
      current_text
    """
    if device_str != ov_model._device:
        ov_model.request = None
        ov_model._device = device_str

        for i in progress.tqdm(range(1), desc=f"Model loading on {device_str}"):
            ov_model.compile()
    return current_text

available_devices = Core().available_devices + ["AUTO"]

examples = [
    "Give me recipe for pizza with pineapple",
    "Write me a tweet about new OpenVINO release",
    "Explain difference between CPU and GPU",
    "Give five ideas for great weekend with family",
    "Do Androids dream of Electric sheep?",
    "Who is Dolly?",
    "Please give me advice how to write resume?",
    "Name 3 advantages to be a cat",
    "Write instructions on how to become a good AI engineer",
    "Write a love letter to my best friend",
]
#----------------------------------------------------------------------------------------------------
#socket programing

def run_server():
    # Create a socket object
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Bind the socket to a port
    host = "localhost"
    port = 8080
    sock.bind((host, port))

    # Listen for connections
    sock.listen(5)

    # Accept a connection
    conn, addr = sock.accept()

    # Get the prompt from the client
    prompt = conn.recv(1024).decode()
    print("this is the prompt", prompt)
    top_p=0.92
    top_k=0
    temperature=0.8
    max_new_tokens=512
    perf_text="test"

    # Generate text
    #generated_text = run_generation(prompt, top_p=0.92, temperature=0.8, top_k=0, max_new_tokens=512, performance="test")
    
    generated_text = run_generation(prompt, top_p, temperature, top_k, max_new_tokens, perf_text)

    print("                                                        ")
    print("                                                        ")
    print("                                                        ")
    print("this is the generated_text", generated_text)
    print("                                                        ")
    print("                                                        ")
    print("                                                        ")
    print("this is the type generated_text", type(generated_text))
    print("                                                        ")
    print("                                                        ")
    print("                                                        ")
    print("this is the generated text", list(generated_text)) 
    print("                                                        ")
    print("                                                        ")
    print("                                                        ")
    print("                                                        ")
    print("                                                        ")

    # Stream the output to the client
    while generated_text:
        #generated_text = run_generation(prompt, top_p, temperature, top_k, max_new_tokens, perf_text)
        conn.sendall(generated_text.encode())

    # Close the connection
    conn.close()

run_server()
#------------------------------------------------------------------------------------------------------

'''
with gr.Blocks() as demo:
    gr.Markdown(
        "# Instruction following using Databricks Dolly 2.0 and OpenVINO.\n"
        "Provide insturction which describes a task below or select among predefined examples and model writes response that performs requested task."
    )

    with gr.Row():
        with gr.Column(scale=4):
            user_text = gr.Textbox(
                placeholder="Write an email about an alpaca that likes flan",
                label="User instruction"
            )
            model_output = gr.Textbox(label="Model response", interactive=False)
            performance = gr.Textbox(label="Performance", lines=1, interactive=False)
            with gr.Column(scale=1):
                button_clear = gr.Button(value="Clear")
                button_submit = gr.Button(value="Submit")
            gr.Examples(examples, user_text)
        with gr.Column(scale=1):
            device = gr.Dropdown(choices=available_devices, value=current_device, label="Device")
            max_new_tokens = gr.Slider(
                minimum=1, maximum=1000, value=256, step=1, interactive=True, label="Max New Tokens",
            )
            top_p = gr.Slider(
                minimum=0.05, maximum=1.0, value=0.92, step=0.05, interactive=True, label="Top-p (nucleus sampling)",
            )
            top_k = gr.Slider(
                minimum=0, maximum=50, value=0, step=1, interactive=True, label="Top-k",
            )
            temperature = gr.Slider(
                minimum=0.1, maximum=5.0, value=0.8, step=0.1, interactive=True, label="Temperature",
            )

    user_text.submit(run_generation, [user_text, top_p, temperature, top_k, max_new_tokens, performance], [model_output, performance])
    button_submit.click(select_device, [device, user_text], [user_text])
    button_submit.click(run_generation, [user_text, top_p, temperature, top_k, max_new_tokens, performance], [model_output, performance])
    button_clear.click(reset_textbox, [user_text, model_output, performance], [user_text, model_output, performance])
    device.change(select_device, [device, user_text], [user_text])
'''
if __name__ == "__main__":
    try:
        run_server()
        #demo.launch(enable_queue=True, share=True, height=800)
    except Exception:
        print("throwing exception")
        #demo.launch(enable_queue=True, share=True, height=800)
