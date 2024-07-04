import re, os, yaml
from ai_core import APP_DIR
from loguru import logger as log
import requests
from urllib.parse import urljoin

model_templates_config_fp = os.path.join(APP_DIR, "model_templates.yaml")

def read_config():
    with open(model_templates_config_fp, "r", encoding="utf-8") as f:
        return yaml.safe_load(f.read())

def tokenize(*args):
    """
    Accepts multple arguments similar to `print` function, and tokenizes resulting text.
    """
    from nltk.tokenize import word_tokenize
    text = " ".join([str(arg) for arg in args])
    tokens = word_tokenize(text)
    return tokens

def untokenize(tokens):
    from nltk.tokenize.treebank import TreebankWordDetokenizer
    text = TreebankWordDetokenizer().detokenize(tokens)
    return text


def trim_text_by_tokens(text, max_tokens, from_start=True):
    """Trims text to fit into max_tokens length.

    Args:
        text (str): Input text
        max_tokens (int): Maximum amount of tokens
        from_start (bool, optional): Where to start cutting text from. If True - will cut from beginning, leaving the ending part of the text. If False will cut from beginning. Defaults to True.

    Returns:
        _type_: _description_
    """
    if isinstance(text, list):
        text = " ".join(text)
    text = text.replace("\n", r"<llbr>")
    tokens = tokenize(text)
    if len(tokens) <= max_tokens:
        return text.replace("<llbr>", "\n").replace("llbr>", "\n").replace("<llbr", "\n")

    # Adding one just in case to make sure we are UNDER the limit. Maybe it's rudimentary.
    trim_tokens_count = len(tokens) - max_tokens + 1

    if from_start:
        trimmed = untokenize(tokens[trim_tokens_count:])
    else:
        trimmed = untokenize(tokens[:len(tokens) - trim_tokens_count])
    trimmed = str(trimmed).replace("<llbr>", "\n").replace("llbr>", "\n").replace("<llbr", "\n")
    return trimmed

def count_tokens_nltk(*args):
    return len(tokenize(*args))




def trim_incomplete_sentence(txt):
    # Cache length of text
    ln = len(txt)
    # Find last instance of punctuation (Borrowed from Clover-Edition by cloveranon)
    lastpunc = max(txt.rfind("."), txt.rfind("!"), txt.rfind("?"))
    # Is this the end of a quote?
    if(lastpunc < ln-1):
        if(txt[lastpunc+1] == '"'):
            lastpunc = lastpunc + 1
    if(lastpunc >= 0):
        txt = txt[:lastpunc+1]
    return txt

def get_config_from_model_name(model_name):
    model_templates_config = read_config()
    for regex, model_config in model_templates_config.items():
        if re.match(regex, model_name, re.IGNORECASE):
            log.debug(
                f"Found matching model config: {model_name} (REGEX: {regex})"
            )
            return model_config
    raise Exception(f"Unable to find matching config for model: {model_name}")

def get_prompt_format_from_model_name(model_name):
    model_config = get_config_from_model_name(model_name)
    return model_config['prompt_format']


def extract_urls(text, base_site=None):
    regex = r"(?:|\()\/Content\/Attachments\/.*?(?: |\))"
    urls = re.findall(regex, text)
    print(urls)
    for url in urls:
        text = text.replace(url, "")
    regex = r"(?:http[s]?:\/\/.)?(?:www\.)?[-a-zA-Z0-9@%._\+~#=]{2,256}\.[a-z]{2,6}\b(?:[-a-zA-Z0-9@:%_\+.~#?&\/\/=]*)"
    urls.extend(re.findall(regex, text))
    print(urls)
    urls = [url.replace("(", "").replace(")", "").replace("\n", "") for url in urls]
    extracted_urls = []
    if base_site is not None:
        for url in urls:
            if url.startswith(("http://", "https://")):  # Handle URLs starting with http or https
                extracted_urls.append(url)
            else:  # Handle relative URLs
                full_url = urljoin(base_site, url)
                extracted_urls.append(full_url)
    else:
        extracted_urls = urls
    print(extracted_urls, urls)
    return extracted_urls, urls


def is_image_url(url):
    print(f"Testing if {url} is image...")
    try:
        response = requests.head(url, timeout=30)
        if not response:
            return False
        content_type = response.headers.get('content-type')
        # print(conten)
        if content_type is not None and 'image' in content_type:
            return True
        return False
    except Exception as e:
        print(e)
        return False
    
def extract_image_urls(text, base_site=None):
    """
    Raw urls are urls as they appear in text
    urls are absolute urls, which were converted if they were not absolute"""
    image_urls = []
    raw_image_urls = []
    non_image_urls = []
    raw_non_image_urls = []
    urls, raw_urls = extract_urls(text, base_site=base_site)
    for i, url in enumerate(urls):
        if is_image_url(url):
            image_urls.append(url)
            raw_image_urls.append(raw_urls[i])
        else:
            non_image_urls.append(url)
            raw_non_image_urls.append(raw_urls[i])
    return image_urls, raw_urls, non_image_urls, raw_non_image_urls