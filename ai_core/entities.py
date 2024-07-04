import random
from loguru import logger as log

class RPCharacter():
    def __init__(self, *args, **kwargs):
        self.name = kwargs.get("name", "CharacterName")
        self.description = kwargs.get("description", "")
        self.traits = kwargs.get("traits", "")
        self.gender = kwargs.get("gender", "")
        self.summary = kwargs.get("summary", "")
        self.tasks = kwargs.get("tasks", [])
        self.world_info = kwargs.get("world_info", "")
        self.examples = kwargs.get("examples", "")
        self.species = kwargs.get("species", "")
        self.height = kwargs.get("height", "")
        self.weight = kwargs.get("weight", "")
        self.roles = kwargs.get("roles", "")
        self.age = kwargs.get("age", "")
        self.first_message = kwargs.get("first_message", "")

        # This is CURRENT context knowledge
        self.knowledge = None

        # These are loaded from config
        self.triggerword_knowledge = []
        self.fuzzy_knowledge = []
        if self.name == "CharacterName":
            log.warning("Character name is not supplied.")

    @property
    def task(self):
        if self.tasks:
            return self.tasks[0]
        return None

    def sample_examples(self, n):
        if not self.examples:
            return []
        if len(self.examples) < n:
            n = len(self.examples)
        return random.sample(self.examples, n)

