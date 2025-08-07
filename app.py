from fastapi import FastAPI, Body
from pydantic import BaseModel
from typing import Optional
import uvicorn
import uuid
from dotenv import load_dotenv

from src.main import HelpCentreAssistant

load_dotenv()


class QuestionRequest(BaseModel):
    """Request model for asking a question."""
    question: str
    thread_id: Optional[str] = None


app = FastAPI(
    title="Help Centre Assistant API",
    description="API for interacting with the Help Centre Assistant.",
    version="1.0.0",
)

assistant = HelpCentreAssistant()

@app.post("/ask_question")
def ask_question(request: QuestionRequest = Body(...)):
    """
    Handles a user's question to the Help Centre Assistant.

    - **question**: The question to be answered.
    - **thread_id**: The ID of the conversation thread. If not provided, a new one is generated.
    """
    thread_id = request.thread_id if request.thread_id else str(uuid.uuid4())
    
    answer = assistant.run(question=request.question, thread_id=thread_id)
    
    return {
        "answer": answer,
        "thread_id": thread_id
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80)

