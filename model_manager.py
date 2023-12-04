from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from streamer import TokenStreamer
import torch
import os
import json
import multiprocessing as mp
import time

class LLM: # TODO: create base class
    """
    Runs in a separate process and only interacts with the ModelManager.
    """
    def __init__(self, output_queue):
        self.model = None
        self.tokenizer = None
        self.output_queue = output_queue # anything put in this queue will be displayed in the interface
        self.device_map = "cuda"
    
    def inference(self, input):
        """
        Inferences the model using a token streamer which will stream tokens to output queue
        """
        if self.model is not None and self.tokenizer is not None:
            streamer = TokenStreamer(self.tokenizer, self.output_queue, skip_prompt=True)

            generation_config = GenerationConfig(
                temperature = input['hyperparameters']['temperature'],
                top_p = input['hyperparameters']['top_p'],
                top_k = input['hyperparameters']['top_k'],
                max_new_tokens = input['hyperparameters']['max_new_tokens'],
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

            # All the device map values except "auto" can be used by pytorch to send tensors
            if self.device_map == "auto":
                # This check isn't foolproof since a model loaded with device_map="auto"
                # can potentially be split between the GPU and CPU
                # TODO: better way to determine the "model_inputs.to(device)"
                if torch.cuda.is_available():
                    model_inputs = self.tokenizer([input['text']], return_tensors="pt").to("cuda")
                else:
                    model_inputs = self.tokenizer([input['text']], return_tensors="pt").to("cpu")
            else:
                model_inputs = self.tokenizer([input['text']], return_tensors="pt").to(self.device_map)

            self.model.generate(**model_inputs, streamer=streamer, generation_config=generation_config)

    def load_model(self, model_dir, device_map): # TODO: handle CUDA out of memory error
        """
        Loads a given model (and tokenizer) to the provided device
        """
        self.device_map = device_map
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, device_map=device_map)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, device_map=device_map)

class ModelManager():
    """
    Manages the LLM process and handles communication between the interface and the LLM
        - LLM must be run in seprate process to enable loading and unloading models without
        restarting the interface (cuda context doens't release memory until process terminates)
    """
    def __init__(self, 
                 model, 
                 device_map, 
                 temperature,
                 top_p,
                 top_k,
                 max_new_tokens,
                 inference_speed_callback, 
                 context_length_callback, 
                 token_output_callback, 
                 model_loaded_callback,
                 model_unloaded_callback
                 ):
        
        self.model_dir = os.path.join("./models", model)
        self.device_map = device_map
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        self.inference_speed_callback = inference_speed_callback
        self.context_length_callback = context_length_callback
        self.token_output_callback = token_output_callback
        self.model_loaded_callback = model_loaded_callback
        self.model_unloaded_callback = model_unloaded_callback
        
        with open(os.path.join(self.model_dir, "config.json"), 'r') as model_config:
            data = json.load(model_config)
            self.max_context_length = data.get("sliding_window", None)

        self.model_process = None
        self.model_ready_event = mp.Event()
        self.worker_running_event = mp.Event()
        self.inference_running_event = mp.Event()
        self.input_queue = mp.Queue()
        self.output_queue = mp.Queue()

        self.output_started = False

    def run_inference(self, text):
        """
        Sends text to model process for inference, waits for inference to start,
        then passes tokens and stats from output queue to interface callbacks
        """
        if self.model_ready_event.is_set():
            input = dict(
                text=text,
                hyperparameters=dict(
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    max_new_tokens=self.max_new_tokens
                )
            )
            self.input_queue.put_nowait(input)
            self.inference_running_event.wait()

            while self.inference_running_event.is_set():
                if not self.output_queue.empty():
                    self.output_started = True
                    output = self.output_queue.get()
                    self.inference_speed_callback(output['inference_speed'])
                    self.context_length_callback(output['context_used'], self.max_context_length)
                    self.token_output_callback(output['token_output'])
            
            self.output_started = False

    def model_worker(self):
        """
        The entry point for the model processes
        """
        llm = LLM(self.output_queue)
        llm.load_model(self.model_dir, self.device_map)
        self.model_ready_event.set()

        while self.worker_running_event.is_set():
            if not self.input_queue.empty():
                self.inference_running_event.set()
                input = self.input_queue.get()
                llm.inference(input)
                self.inference_running_event.clear()
            else:
                time.sleep(0.1)

    def start_model_worker(self):
        if self.model_process is None:
            self.worker_running_event.set()
            self.model_process = mp.Process(target=self.model_worker)
            self.model_process.start()
            self.model_ready_event.wait()
            self.model_loaded_callback()

    def stop_model_worker(self):
        if self.model_process is not None and self.model_process.is_alive():
            self.worker_running_event.clear()
            self.model_process.terminate()
            self.model_process.join()
            self.model_process = None
            self.model_ready_event.clear()
            self.model_unloaded_callback()

    def set_hyperparameters(self, settings):
        self.llm_parameters = settings

    def set_model(self, model):
        """
        Sets the model directory, and checks the config.json for the max context length
        """
        self.model_dir = os.path.join("models", model, "config.json")
        with open(self.model_dir, 'r') as model_config:
            data = json.load(model_config)
            self.max_context_length = data.get("sliding_window", None)

