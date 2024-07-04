from typing import List, Optional
from pydantic import BaseModel, PrivateAttr
from cachetools import TTLCache, cached
import requests
from copy import copy
from loguru import logger as log
from ai_core.integrations import ChatAPI, CompletionAPI
from ai_core.memory import Message, AIMessage, SystemMessage, UserMessage
from cachetools.keys import hashkey


class OpenedAIException(Exception):
    pass

class _OpenedAICommonMixin(BaseModel):
    host: str
    _base_url: str = PrivateAttr()
    _completions_url: str = PrivateAttr()
    _chat_completions_url: str = PrivateAttr()
    _model_info_url: str = PrivateAttr()
    _tokens_url: str = PrivateAttr()
    
    def __init__(self, **data):
        super().__init__(**data)

        # TODO Use urljoin
        self._base_url = f'{self.host}/v1'
        self._completions_url = f'{self._base_url}/completions'
        self._chat_completions_url = f'{self._base_url}/chat/completions'
        self._model_info_url = f'{self._base_url}/internal/model/info'
        self._tokens_url = f'{self._base_url}/internal/token-count'

    def get_allowed_payload_keys(self):
        log.warning("Not implemented on base mixin class. Wrong function used.")
        raise NotImplementedError("Not implemented on base mixin class. Wrong function used.")

    def filter_payload_by_schema(self, payload):
        """
        Remove keys that are not allowed, based on swagger api schema available in ooba
        """
        remove_keys = [key for key in payload.keys() if key not in self.get_allowed_payload_keys()]
        for key in remove_keys:
            payload.pop(key, None)

        return payload
    
    def __key(self):
        return (self.host)

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__key() == other.__key()
        return NotImplemented
    
    @property
    def loaded_model(self):
        return requests.get(self._model_info_url).json()['model_name']

class OpenedAI(_OpenedAICommonMixin, CompletionAPI):
    def __call__(self, prompt, parameters={}, system_message=""):
        super().__call__(prompt, parameters, system_message)

        payload = {
            "prompt": prompt,
            "stream": False
        }

        payload.update(parameters)

        payload = self.filter_payload_by_schema(payload)
        response = requests.post(self._completions_url, json=payload, headers={"Content-Type": "application/json"})

        if response.status_code != 200:
            log.warning(f"ERROR: {response.status_code} {response.text}")
            raise OpenedAIException(f"Error response from the server ({self._completions_url}): {response.status_code} ({response.text})")

        response_text = response.json()['choices'][0]['text']

        log.debug(f"AI:\n{response_text}")
        log.debug(f"Finish reason: {response.json()['choices'][0]['finish_reason']}")
        return response_text
    
    @cached(cache=TTLCache(maxsize=4, ttl=3600))
    def get_allowed_payload_keys(self):
        openapi_json = requests.get(self.host + "/openapi.json").json()
        allowed_keys = openapi_json['components']['schemas']['CompletionRequest']['properties'].keys()
        return allowed_keys

class OpenedAIChat(_OpenedAICommonMixin, ChatAPI):
    def __call__(self, messages, parameters={}):
        super().__call__(messages, parameters)
        payload = dict(
            messages = self.convert_messages_to_openedai_dict(messages=messages)
        )
        payload.update(parameters)

        payload = self.filter_payload_by_schema(payload)
        response = requests.post(self._chat_completions_url, json=payload, headers={"Content-Type": "application/json"})

        if response.status_code != 200:
            log.warning(f"ERROR: {response.status_code} {response.text}")
            raise OpenedAIException(f"Error response from the server ({self._chat_completions_url}): {response.status_code} ({response.text})")
        
        response_message = self.convert_openedai_message_to_message(response.json()['choices'][0]['message'])

        log.debug(f"AI:\n{response_message}")
        log.debug(f"Finish reason: {response.json()['choices'][0]['finish_reason']}")
        return response_message
    
    def convert_openedai_message_to_message(self, message:dict):
        if message['role'] == "assistant":
            message = AIMessage(text=message['content'])
        elif message['role'] == "system":
            message = SystemMessage(text=message['content'])
        elif message['role'] == "assistant":
            message = UserMessage(text=message['content'])
        return message

    def convert_messages_to_openedai_dict(self, messages:List[Message]):
        openedai_messages = []
        for message in messages:
            if isinstance(message, SystemMessage):
                role = "system"
            elif isinstance(message, AIMessage):
                role = "assistant"
            else:
                role = "user"
            openedai_messages.append(
                dict(
                    role = role,
                    content = message.text
                )
            )
        return openedai_messages

    @cached(cache=TTLCache(maxsize=4, ttl=30))
    def get_allowed_payload_keys(self):
        openapi_json = requests.get(self.host + "/openapi.json").json()
        allowed_keys = list(openapi_json['components']['schemas']['ChatCompletionRequest']['properties'].keys())
        return allowed_keys

