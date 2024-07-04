# AI Core

A simple yet effective toolset for working with LLMs.

If you think that LangChain is bloated and unnecessarily complex and hard to work with - this might be for you.

This module, however, is purpose built for my roleplay chatbots, and as of right now I am not planning on offering wide support for it. However the project is open for pull requests.

## Features

- Supports local LLM providers and OpenAI through "OpenedAI API"
- Can be easily expanded with more LLM provider APIs if necessary
- Powerful templating with jinja
- Automatic detection of currently loaded LLM model, and selection of correct instruction-chat templates
- Support for System, AI, User message types, and automatic formatting of chat logs
- Support for various chat memory systems, like rolling window
    - Support for memory with summarization is planned
- Support for integrating chat systems with vision models, automatic image URL detection and replacing it with BLIP captions (local), WD1.4 tags (remote, Automatic1111 API) or vLLM captions (remote, OLLAMA API)
- Probably more

# Example usage


```python
from ai_core import presets
from ai_core.integrations.openedai import OpenedAIChat, OpenedAI
from ai_core.memory import (
    AIMessage,
    Memory,
    Memory_SlidingWindow,
    Message,
    SystemMessage,
    UserMessage,
    messages_to_plaintext
)
from ai_core.templating import messages_to_prompt, render_template_string, SystemMessageTemplate
from ai_core.utils import vision, trim_incomplete_sentence, get_config_from_model_name


completion = OpenedAI(host='http://192.168.1.20:5000')
template_name = utils.get_prompt_format_from_model_name(model_name)

messages = [
    SystemMessage("You are an assistant"),
    UserMessage("Hello who are you?")
]

prompt = messages_to_prompt(messages, next_message_name=data['bot_name'], template_name=template_name)
parameters = presets.load_preset(preset)
    parameters['stop'] = ["<|im_end|>", "|im_end|", "<s>", "</s>", "<|eot_id|>", "<|start_header_id|>", "assistant\n", "assistant>\n", "assistant|>\n"]
    parameters['max_tokens'] = 2000
response = completion(
        prompt=str(prompt),
        parameters=parameters
            )
```

## Using memory systems

```python

memory = Memory_SlidingWindow(token_limit=8000) # There are tools to get token limit for currently loaded model from config files
memory.add_message(SystemMessage("You are an assistant"))
memory.add_message(UserMessage("Hello who are you?"))

messages = memory.messages

```