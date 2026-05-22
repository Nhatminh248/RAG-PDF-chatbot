import os
import pypdf
from typing import List, Any
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder

# Load environment variables
load_dotenv()

# Persist directory configuration
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "pdf_rag_collection"

# Initialize local embeddings (all-MiniLM-L6-v2) - ~90MB download on first run
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize ChromaDB vector store
db = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME
)

# Global holder for lazy-loaded CrossEncoder
_reranker_instance = None

def _get_cross_encoder():
    global _reranker_instance
    if _reranker_instance is None:
        # Load cross-encoder/ms-marco-MiniLM-L-6-v2 for semantic reranking
        _reranker_instance = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker_instance

class HybridRerankRetriever(BaseRetriever):
    db: Any

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        # 1. Fetch all documents from persistent store to construct BM25 retriever dynamically
        res = self.db.get()
        if not res or not res["documents"]:
            return []
            
        docs = []
        for text, meta in zip(res["documents"], res["metadatas"]):
            docs.append(Document(page_content=text, metadata=meta))
            
        # 2. Setup dynamic BM25Retriever
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = 10  # Retrieve top 10
        
        # 3. Setup standard Vector Store retriever
        vector_retriever = self.db.as_retriever(search_kwargs={"k": 10})
        
        # 4. Ensemble both search paradigms with equal weights
        ensemble = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.5, 0.5]
        )
        
        # 5. Retrieve top-10 chunks from hybrid ensemble
        hybrid_docs = ensemble.invoke(query)[:10]
        if not hybrid_docs:
            return []
            
        # 6. Rerank using CrossEncoder and keep top-3
        cross_encoder = _get_cross_encoder()
        pairs = [(query, doc.page_content) for doc in hybrid_docs]
        scores = cross_encoder.predict(pairs)
        
        scored_docs = sorted(zip(hybrid_docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored_docs[:3]]

def ingest_pdf(file, filename) -> bool:
    """
    Parses a PDF file-like object, splits it into chunks, and uploads to ChromaDB.
    Returns True if successfully ingested, False if already indexed.
    """
    # Check if already indexed
    if is_pdf_indexed(filename):
        return False
        
    try:
        # Read PDF pages using pypdf
        reader = pypdf.PdfReader(file)
        documents = []
        
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                # Store page number (1-indexed) and source filename
                metadata = {
                    "source": filename,
                    "page": i + 1
                }
                documents.append(Document(page_content=text, metadata=metadata))
        
        if not documents:
            raise ValueError(f"Could not extract any text from '{filename}'. It might be scanned or empty.")
            
        # Split documents using RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        
        # Add chunks to vector store
        db.add_documents(chunks)
        return True
        
    except Exception as e:
        raise RuntimeError(f"Error parsing PDF '{filename}': {str(e)}")

def is_pdf_indexed(filename: str) -> bool:
    """
    Checks if a PDF has already been indexed in ChromaDB.
    """
    res = db.get(where={"source": filename}, limit=1)
    return len(res.get("ids", [])) > 0

def get_indexed_pdfs() -> list:
    """
    Retrieves a list of all unique PDF filenames currently indexed in the vector store.
    """
    res = db.get()
    metadatas = res.get("metadatas", [])
    if not metadatas:
        return []
    
    # Extract unique source names
    sources = set()
    for meta in metadatas:
        if meta and "source" in meta:
            sources.add(meta["source"])
            
    return sorted(list(sources))

def delete_pdf(filename: str) -> bool:
    """
    Deletes all chunks belonging to a specific PDF from ChromaDB.
    """
    res = db.get(where={"source": filename})
    if res and res["ids"]:
        db.delete(ids=res["ids"])
        return True
    return False

def delete_all() -> bool:
    """
    Clears all documents from the ChromaDB collection.
    """
    res = db.get()
    if res and res["ids"]:
        db.delete(ids=res["ids"])
        return True
    return False

def get_answer(query: str, chat_history: list) -> dict:
    """
    Queries the vector store using a conversational retrieval chain.
    Returns a dictionary containing the answer and source documents.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set. Please set it in your environment or .env file.")
        
    # Set up OpenRouter LLM using langchain-openai
    llm = ChatOpenAI(
        model="google/gemini-2.0-flash-exp:free",
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.3,
        # OpenRouter requires these extra headers sometimes, though basic works
        default_headers={
            "HTTP-Referer": "https://github.com/langchain-ai/langchain", 
            "X-Title": "RAG PDF Chatbot",
        }
    )
    
    # Set up hybrid vector/BM25 retriever with semantic CrossEncoder reranking
    retriever = HybridRerankRetriever(db=db)
    
    # Create the ConversationalRetrievalChain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    
    # Get response from LangChain
    response = chain.invoke({
        "question": query,
        "chat_history": chat_history
    })
    
    return {
        "answer": response["answer"],
        "source_documents": response.get("source_documents", [])
    }
