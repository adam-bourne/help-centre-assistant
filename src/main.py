from langchain_core.messages import HumanMessage

from src.graph import create_rag_graph


class HelpCentreAssistant:
    """Main class for the Help Centre Assistant"""

    def __init__(self):
        """Initialize the Help Centre Assistant"""

        self.graph = create_rag_graph()
        self.thread_store = {}

    def run(self, question: str, thread_id: str):
        """Run the Help Centre Assistant"""

        if thread_id not in self.thread_store:
            messages = [HumanMessage(content=question)]
        else:
            messages = self.thread_store[thread_id]
            messages.append(HumanMessage(content=question))

        response = self.graph.invoke({"messages": messages})

        self.thread_store[thread_id] = response["messages"]

        final_response = response["messages"][-1].content

        return final_response