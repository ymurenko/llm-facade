import textwrap

def wrap_text(text, width):
    """
    DIY text-wrapping solution for DearPyGui, which doens't
    support text-wrpping inside of text input areas.
        - Not ideal, since the copied text will have extra newlines
    """
    lines = text.split('\n')
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    return '\n'.join(wrapped_lines)

class Conversation:
    """
    Manages the LLM conversation and handles formatting
    """
    def __init__(self, system_prompt, instruction_start_string, instruction_end_string):
        self.instruction_start_string = instruction_start_string # TODO: instruction strings should be set by user
        self.instruction_end_string = instruction_end_string
        self.show_system_prompt = False
        self.show_raw_output = False
        self.line_width = 120
        self.conversation_history = [
            {
                "role": "system",
                "content": system_prompt,
            },
        ]

    def remove_last_message(self):
        """
        Removes last message from conversation history
        """
        if(len(self.conversation_history) > 1):
            self.conversation_history.pop(-1)

    def reset_conversation(self):
        """
        Resets conversation history
        """
        self.conversation_history = [
            {
                "role": "system",
                "content": self.conversation_history[0]["content"],
            },
        ]

    def append_full_user_message(self, message):
        """
        Adds user message to conversation history
        """
        self.conversation_history.append({
            "role": "user",
            "content": message,
        })

    def append_partial_ai_message(self, text):
        """
        Appends tokens to the last AI message in the conversation history
        """
        if(self.conversation_history[-1]["role"] == "user"):
            self.conversation_history.append({
                "role": "ai",
                "content": "",
            })

        self.conversation_history[-1]["content"] += text

    def set_system_prompt(self, message):
        self.conversation_history[0]["content"] = message

    def get_system_prompt(self):
        return self.conversation_history[0]["content"]

    def get_string_for_inference(self):
        """
        Parses conversation history into a raw string that can be inferenced
        """
        input_string = self.conversation_history[0]["content"] + "\n\n"
        for message in self.conversation_history[1:]:
            if message["role"] == "user":
                input_string += self.instruction_start_string + " " + message["content"] + " " + self.instruction_end_string + "\n\n"
            elif message["role"] == "ai":
                input_string += message["content"] + "\n\n"

        return input_string
    
    def get_string_for_display(self):
        """
        Parses conversation history into a neat string to display
        in the interface's chat output area, returns raw string if
        show_raw_output is set using the chekcbox in the interface
        """
        output_string = ""
        
        if self.show_raw_output:
            output_string = self.get_string_for_inference()

        else:
            if(self.show_system_prompt):
                output_string = "<SYSTEM INSTRUCTIONS>: " + wrap_text(self.conversation_history[0]["content"], width=self.line_width) + "\n\n"

            for message in self.conversation_history:
                if message["role"] == "user":
                    output_string += "<USER>: " + wrap_text(message["content"], width=self.line_width) + "\n\n"
                elif message["role"] == "ai":
                    output_string += "<AI>: " + wrap_text(message["content"], width=self.line_width) + "\n\n"

        return output_string