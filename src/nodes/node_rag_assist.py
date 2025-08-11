import os
from pathlib import Path
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from sentence_transformers import SentenceTransformer
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from src.models.state import State
from src.constants import (
    INDEX_NAME, ALPHA, RETRIEVAL_TOP_N, RERANKER_TOP_N, RERANKER_MODEL,
    SENTENCE_TRANSFORMER)

# Initialize Pinecone
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Load models
dense_model = SentenceTransformer('all-MiniLM-L6-v2')
project_root = Path(__file__).parent.parent.parent
sparse_model_path = str(project_root / "data" / "bm25_values.json")
sparse_model = BM25Encoder().load(sparse_model_path)


def node_rag_assist(state: State) -> Command:
    """Node for the RAGAssist tool"""

    # Get the query from the last message
    tool_call = state["messages"][-1].tool_calls[0]
    query = tool_call["args"]["query"]

    search_results = hybrid_search(query)
    reranked_results = rerank_results(search_results, query)

    formatted_chunks = (
        "Here are the most relevant chunks of information for the user's query:\n"
        + ''.join([
            f"---------------Chunk {doc['index']}---------------\n"
            f"{doc['document']['text']}\n"
            f"---------------------------------------\n"
            for doc in reranked_results.data
        ])
        + "\nUse this information to answer the user's question"
    )

    # Create the tool message
    tool_message = ToolMessage(
        content=formatted_chunks,
        tool_call_id=tool_call["id"]
    )

    return Command(
        update={"messages": [tool_message]},
        goto="node_main_assistant"
    )


def hybrid_search(query: str, top_k: int = RETRIEVAL_TOP_N, alpha: float = ALPHA):
    """
    Perform hybrid search using both dense and sparse vectors
    alpha: 0.0 = sparse (BM25) only, 1.0 = dense only, 0.5 = equal weight
    """
    if not 0 <= alpha <= 1:
        raise ValueError("Alpha must be between 0 and 1")
        
    # Get dense vector for query
    dense_query = dense_model.encode(query).tolist()
    
    # Get sparse vector for query
    sparse_query = sparse_model.encode_queries(query)
    
    hdense, hsparse = hybrid_score_norm(dense_query, sparse_query, alpha)
    
    # Perform hybrid search
    results = index.query(
        vector=hdense,
        sparse_vector=hsparse,
        top_k=top_k,
        include_metadata=True
    )
    
    return results


def rerank_results(results: dict, query: str, top_k: int = RERANKER_TOP_N):
    """Rerank the results using the reranker model"""
    
    documents = [
        {"id": x["id"], "text": x["metadata"]["text"]} 
        for x in results["matches"]
    ]

    reranked_documents = pc.inference.rerank(
        model=RERANKER_MODEL,
        query=query,
        documents=documents,
        top_n=top_k,
        return_documents=True,
        parameters={
            "truncate": "END"
        }
    )
    
    return reranked_documents

def hybrid_score_norm(dense, sparse, alpha: float):
    """Hybrid score using a convex combination

    alpha * dense + (1 - alpha) * sparse

    Args:
        dense: Array of floats representing
        sparse: a dict of `indices` and `values`
        alpha: scale between 0 and 1
    """
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1")
    hs = {
        'indices': sparse['indices'],
        'values':  [v * (1 - alpha) for v in sparse['values']]
    }
    return [v * alpha for v in dense], hs