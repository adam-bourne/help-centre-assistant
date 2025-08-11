from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import Command

from src.models import State, RAGAssist
from src.prompts import MAIN_ASSISTANT_PROMPT
from src.utils import get_openai_llm
from src.constants import ASSISTANT_MODEL

today = datetime.now().strftime("%B %d, %Y")

def node_main_assistant(state: State) -> Command:
    """Main assistant node"""

    llm = get_openai_llm(model=ASSISTANT_MODEL)

    main_assistant_prompt = ChatPromptTemplate.from_messages(
        [("system", MAIN_ASSISTANT_PROMPT),("placeholder", "{messages}")]
    ).partial(
        current_date=today
    )

    main_assistant_runnable = main_assistant_prompt | llm.bind_tools([RAGAssist])

    next_step = main_assistant_runnable.invoke(state)

    return Command(
        update={"messages": [next_step]},  # Wrap in list to ensure we're concatenating lists
        goto="route_next_step"
    )
