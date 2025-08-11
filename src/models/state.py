import operator
from typing import List, TypedDict, Annotated

from langchain_core.messages import BaseMessage

class State(TypedDict):
    """State of the assistant"""

    messages: Annotated[List[BaseMessage], operator.add]