import yaml
import os
from loguru import logger as log
AI_CORE_DIR = os.path.dirname(os.path.realpath(__file__))

def load_preset_string(yaml_string: str):
    return yaml.safe_load(yaml_string)

def load_preset_file(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f.read())
    
def load_preset(preset_name: str):
    filepath = os.path.join(AI_CORE_DIR, "presets", preset_name + ".yaml")
    if not os.path.exists(filepath):
        log.error(f"Preset '{preset_name}' not found at {filepath}!")
        return {}
    return load_preset_file(filepath)