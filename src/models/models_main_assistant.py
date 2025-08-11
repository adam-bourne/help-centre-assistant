from pydantic import BaseModel, Field

class RAGAssist(BaseModel):
    """For searching the Typeform Help Centre documentation"""

    query: str = Field(description="The user's specific question, including context and scope")
