# llm-facade
A python-based text generation app for inferencing large language models, built using [transformers](https://pypi.org/project/transformers/) and [dearpygui](https://pypi.org/project/dearpygui/).

https://github.com/ymurenko/llm-facade/assets/44625754/1ac30b36-ac89-4c5d-ba42-b6dc60e094e6

I've been working with LLMs a lot lately, so I decided to make myself a little app to run a local conversational inference with any of the popular LLMs hosted on hugging face. While projects like [text-generation-webui](https://github.com/oobabooga/text-generation-webui) and [gradio](https://github.com/gradio-app/gradio) are a lot more polished and practical (I built llm-facade v0.1 over a couple weekends lol), they happen to use a web-browser and a modern UI. I want a standalone app that looks and feels a bit more old-school, with a data-rich UI :sunglasses:

Very much a small side project, so it needs a bit more polish as well as testing with different models and system configurations. 

## Features

- Locally load and inference LLMs hosted on [Hugging Face](https://huggingface.co/)
  - Tested with `Mistral-7B-instruct` and `Llama-2-13B` 
- Chat UI with token streaming
- View raw LLM input/output string
- Monitor system performance and tokens per second
- Track context length usage
- Set inference hyperparamters
- Select which hardware to load the LLM on (CPU, GPU, Apple silicon, or auto)
- (WIP) Set custom input string wrappers such as `[INST] ... [/INST]`

## Compatability

- Supports most language models found on Hugging Face that use the `AutoModelForCausalLM` class and require transformers <= 4.34.0.
- Tested and working on Windows 10, should work on Linux and MacOS
  - Includes option to select Apple silicon for model inference, but I haven't tested it so beware of crashes

 ## Requirements

 - Python 3.10
 - CUDA 11.8+
 - Pytorch 2.1.0+

## Running the app

If you don't have a python environment already set up, I recommend first installing [Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html), then creating an environment to run this repo with `conda create -n facade python=3.10`, and finally activating the environment with `conda activate facade`

1. Clone the repo `git clone https://github.com/ymurenko/llm-facade.git`
2. Navigate to repo root `cd llm-facade`
3. Install python packages `pip install -r requirements.txt`
4. Download a model from [Hugging Face](https://huggingface.co/)
5. Place model files into `llm-facade/models/<model name>`
6. Run app with `python interface.py`

## Contributing

Guidelines TBD
