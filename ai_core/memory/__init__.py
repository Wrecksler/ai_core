from typing import List, Optional, Callable
from pydantic import BaseModel
from ai_core.utils import count_tokens_nltk

class Message(BaseModel):
    text: str
    count_tokens_func: Callable = count_tokens_nltk
    name: str = ""

    def __str__(self) -> str:
        return f"{self.__class__.__name__}: {self.text}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}: {self.text}"
    
    @property
    def token_count(self):
        return self.count_tokens_func(str(self))

class SystemMessage(Message):
    def __init__(self, **data):
        super().__init__(**data)
        if not self.name:
            self.name = "System"

class UserMessage(Message):
    def __init__(self, **data):
        super().__init__(**data)
        if not self.name:
            self.name = "User"

class AIMessage(Message):
    def __init__(self, **data):
        super().__init__(**data)
        if not self.name:
            self.name = "AI"

class Memory(BaseModel):
    messages_all: List[Message] = []
    keep_max: int = 0

    def add_message(self, message):
        self.messages_all.append(message)

        if self.keep_max > 0:
            if len(self.messages_all) > self.keep_max:
                self.messages_all.pop(0)

    @property
    def messages(self):
        return self.messages_all
    
    def to_plaintext_all(self):
        return messages_to_plaintext(self.messages_all)
    
    def to_plaintext(self):
        return messages_to_plaintext(self.messages)
    
    def clear(self):
        self.messages_all = []

def messages_to_plaintext(messages):
    text = ""
    for message in messages:
        text += f"{message.name}: {message.text}\n"
    return text

class Memory_SlidingWindow(Memory):
    token_limit: int
    pinned_messages: List[Message] = []
    
    @property
    def messages(self):
        filtered_messages = []
        for msg in self.pinned_messages:
            filtered_messages.append(msg)

        would_be_count = 0

        for message in reversed(self.messages_all):
            would_be_count += message.token_count
            if would_be_count >= self.token_limit:
                break

            filtered_messages.insert(len(self.pinned_messages), message)

        return filtered_messages
    
    def clear(self):
        super().clear()
        self.pinned_messages = []

