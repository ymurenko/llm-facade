# llm-facade
A python-based data rich text generation UI for inferencing large language models, built using [transformers](https://pypi.org/project/transformers/) and [dearpygui](https://pypi.org/project/dearpygui/).

https://github.com/ymurenko/llm-facade/assets/44625754/1ac30b36-ac89-4c5d-ba42-b6dc60e094e6

## Purpose

I wanted to make myself an alternative to [text-generation-webui](https://github.com/oobabooga/text-generation-webui) that runs as a standalone desktop application instead of a web browser. I also wanted to be able to track things like tokens per second, context length limits, and monitor system resources - all within a single compact UI. 

## Features

- Load most LLMs hosted on [Hugging Face](https://huggingface.co/)
  - Tested with `Mistral-7B-instruct` and `Llama-2-13B` 
- Chat UI with token streaming
- View raw LLM input/output string
- Monitor system performance and tokens per second
- Track context length usage
- Set inference hyperparamters
- Select which hardware to load the LLM on (CPU, GPU, Apple silicon, or auto)
- (WIP) Set custom input string wrappers such as `[INST] ... [/INST]`

## Compatability

- Currently supports any language models found on Hugging Face that use the `AutoModelForCausalLM` class (transformers <= 4.34.0).
- Tested and working on Windows 10, should work on Linux and MacOS
  - Includes option to use Apple silicon for inference

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
