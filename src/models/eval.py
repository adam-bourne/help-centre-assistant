from typing import List, Literal
from pydantic import BaseModel, Field, conint, confloat


class QAItem(BaseModel):
    """Single question/answer example with optional evidence spans."""

    question: str
    answer: str
    evidence_spans: List[str] = Field(default_factory=list)


class QAList(BaseModel):
    """Container for multiple QAItem objects used in structured LLM output."""

    items: List[QAItem]


class UnanswerableQuestion(BaseModel):
    """A single unanswerable question synthesized from context."""

    question: str


class RelevanceJudgement(BaseModel):
    """Relevance score for a candidate chunk to a question on a 0-3 integer scale."""

    grade: conint(ge=0, le=3)


class AnswerEval(BaseModel):
    """LLM-judge rubric for end answers."""

    correctness: confloat(ge=0.0, le=1.0)
    faithfulness: confloat(ge=0.0, le=1.0)
    relevance: confloat(ge=0.0, le=1.0)
    verdict: Literal["pass", "fail"]

