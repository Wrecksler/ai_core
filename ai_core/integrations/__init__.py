from typing import List
from ai_core.memory import Message, messages_to_plaintext
from loguru import logger as log

class CompletionAPI:
    def __call__(self, prompt:str, parameters={}, system_message=""):
        log.debug(f"{self.__class__.__name__} request prompt:\n{prompt}")
        return ""
    
    @property
    def loaded_model(self):
        raise NotImplementedError()

class ChatAPI:
    def __call__(self, messages:List[Message], parameters={}):
        log.debug(f"{self.__class__.__name__} request messages:\n{messages_to_plaintext(messages)}")
        return ""
    
    @property
    def loaded_model(self):
        raise NotImplementedError()
