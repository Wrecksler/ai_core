import os
from ai_core import APP_DIR
from jinja2 import Environment, BaseLoader
from ai_core.memory import Message, SystemMessage, UserMessage, AIMessage
from typing import List
from loguru import logger as log
import yaml

def render_template_string(template_string, context, ignore_errors=False):
    try:
        env = Environment(autoescape=False, loader=BaseLoader, trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True)
        rtemplate = env.from_string(template_string)
        return rtemplate.render(**context)
    except Exception as e:
        if ignore_errors:
            return template_string
        else:
            raise e

class SystemMessageTemplate():
    """
    Used to create system messages for bots.
    """
    def __init__(self, template_string) -> None:
        self.template_string = template_string

    @classmethod
    def from_template(cls, template_name):
        with open(os.path.join(APP_DIR, "templates", "system", f"{template_name}.jinja"), "r", encoding="utf-8") as f:
            template_string = f.read()
        instance = cls(template_string)
        return instance
    
    def format(self, **context):
        return render_template_string(self.template_string, context=context)
    
# llama3-instruct
# chatml
def messages_to_prompt(messages:List[Message], config=None, template_name='chatml', next_message_name=None):
    if callable(template_name):
        template_name = template_name()
    config_filepath = os.path.join(APP_DIR, "templates", "chat", f"{template_name}.yaml")
    with open(config_filepath, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f.read())

    text = config['prefix']
    for message in messages:
        if isinstance(message, AIMessage):
            t = config['AIMessage']
        elif isinstance(message, UserMessage):
            t = config['UserMessage']
        elif isinstance(message, SystemMessage):
            t = config['SystemMessage']
        else:
            t = f"UNKNOWN MESSAGE TYPE: {str(type(message))}"
        text += render_template_string(t, context={"message": message})
    text += render_template_string(config['suffix'], context={"next_message_name": next_message_name})

    log.debug(f"Messages to Prompt:\n{text}")
    return text