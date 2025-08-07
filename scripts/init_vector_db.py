import json
import os
from pinecone import Pinecone
from pinecone import ServerlessSpec
from sentence_transformers import SentenceTransformer
from pinecone_text.sparse import BM25Encoder
from dotenv import load_dotenv
from src.constants import INDEX_NAME

load_dotenv()

def init_vectordb():
    """Initialize the vector database if it doesn't exist"""
    
    # Check for required environment variables
    required_env_vars = ['PINECONE_API_KEY']
    missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

    # Initialize Pinecone
    print("Initializing Pinecone...")
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    # Check if index already exists
    if INDEX_NAME in pc.list_indexes().names():
        print(f"Index '{INDEX_NAME}' already exists. Skipping initialization.")
        return

    # Load models
    print("Loading models...")
    dense_model = SentenceTransformer('all-MiniLM-L6-v2')
    sparse_model = BM25Encoder().load("data/bm25_values.json")

    # Create index
    print("Creating index...")
    pc.create_index(
        name=INDEX_NAME,
        vector_type="dense",
        dimension=dense_model.get_sentence_embedding_dimension(),
        metric='dotproduct',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

    # Connect to the index
    index = pc.Index(INDEX_NAME)

    # Load data
    print("Loading data...")
    with open('data/chunks.json', 'r') as f:
        knowledge_base = json.load(f)

    # Create vectors and upsert to Pinecone
    print("Creating and upserting vectors...")
    for i, chunk in enumerate(knowledge_base):
        # Create dense vector embedding
        dense_vector = dense_model.encode(chunk).tolist()
        
        # Create sparse vector using BM25
        sparse_vector = sparse_model.encode_documents(chunk)
        
        # Convert sparse vector to Pinecone format
        sparse_values = sparse_vector["values"]
        sparse_indices = sparse_vector["indices"]
        
        # Upsert to Pinecone with both dense and sparse vectors
        index.upsert(
            vectors=[
                {
                    "id": str(i),
                    "values": dense_vector,
                    "sparse_values": {
                        "values": sparse_values,
                        "indices": sparse_indices
                    },
                    "metadata": {
                        "id": str(i),
                        "text": chunk
                    }
                }
            ]
        )

    print(f"Successfully uploaded {len(knowledge_base)} chunks to Pinecone index '{INDEX_NAME}' with hybrid search enabled")

if __name__ == "__main__":
    init_vectordb()