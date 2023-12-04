import time
class TokenStreamer():
    """
    TokenStreamer implementation copied and modified from huggingface/transformers
    instead of printing to stdout, it puts the output in a queue, and calculates
    some useful metrics like inference speed and output length
    """
    def __init__(self, 
                 tokenizer, 
                 output_queue,
                 skip_prompt: bool = False, 
                 **decode_kwargs
                 ):
        
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.decode_kwargs = decode_kwargs

        self.output_queue = output_queue

        self.token_cache = []
        self.print_len = 0
        self.next_tokens_are_prompt = True
        self.token_time = []
        self.start_time = None

        self.total_context_length = 0
        
    def put(self, value):
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("TextStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]
            self.total_context_length += len(value.tolist())
        else:
            self.total_context_length += 1
            if(value[0] == self.tokenizer.eos_token_id or
               value[0] == self.tokenizer.pad_token_id or
               value[0] == self.tokenizer.bos_token_id):
                return

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        # Add the new token to the cache and decodes the entire thing.
        self.token_cache.extend(value.tolist())

        text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)

        # After the symbol for a new line, we flush the cache.
        if text.endswith("\n"):
            printable_text = text[self.print_len :]
            self.token_cache = []
            self.print_len = 0
        # If the last token is a CJK character, we print the characters.
        elif len(text) > 0 and self._is_chinese_char(ord(text[-1])):
            printable_text = text[self.print_len :]
            self.print_len += len(printable_text)
        # Otherwise, prints until the last space char (simple heuristic to avoid printing incomplete words,
        # which may change with the subsequent token -- there are probably smarter ways to do this!)
        else:
            printable_text = text[self.print_len : text.rfind(" ") + 1]
            self.print_len += len(printable_text)
        if self.start_time is not None:
            self.token_time.append(time.time() - self.start_time)
        self.start_time = time.time()
        self.on_finalized_text(printable_text)

    def end(self):
        if len(self.token_cache) > 0:
            text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)
            printable_text = text[self.print_len :]
            self.token_cache = []
            self.print_len = 0
        else:
            printable_text = ""

        self.next_tokens_are_prompt = True
        self.on_finalized_text(printable_text, stream_end=True)

    def on_finalized_text(self, text: str, stream_end: bool = False):
        output = dict(
            token_output = text,
            context_used = self.total_context_length,
            inference_speed = 0.0
        )

        sum_token_time = sum(self.token_time)
        len_token_time = len(self.token_time)
        if(len_token_time > 0 and sum_token_time > 0):
            output["inference_speed"] = 1/(sum_token_time / len_token_time)
        
        self.output_queue.put_nowait(output)

        if stream_end:
            self.token_time = []
    
    def _is_chinese_char(self, cp):
        if (
            (cp >= 0x4E00 and cp <= 0x9FFF)
            or (cp >= 0x3400 and cp <= 0x4DBF)  #
            or (cp >= 0x20000 and cp <= 0x2A6DF)  #
            or (cp >= 0x2A700 and cp <= 0x2B73F)  #
            or (cp >= 0x2B740 and cp <= 0x2B81F)  #
            or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
        ):  #
            return True

        return False