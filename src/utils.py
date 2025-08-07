import os
from langchain_openai import ChatOpenAI

from src.constants import LANGCHAIN_PROJECT

# set up tracing on langsmith if using
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT

# cache to avoiding loading models multiple times
_llm_cache = {}

def get_openai_llm(model: str = "gpt-4.1", temperature: float = 0.0, **kwargs) -> ChatOpenAI:
    """Returns an OpenAI LLM"""

    cache_key = f"{model}-{temperature}-{kwargs}"

    if cache_key not in _llm_cache:
        _llm_cache[cache_key] = ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=os.environ["OPENAI_API_KEY"],
        model_kwargs=kwargs
    )

    return _llm_cache[cache_key]