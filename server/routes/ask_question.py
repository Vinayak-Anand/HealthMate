from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse

from modules.llm import get_llm_chain
from modules.query_handlers import query_chain

from langchain_core.documents import Document
from langchain.schema import BaseRetriever

from pinecone import Pinecone
from pydantic import Field
from typing import List
from logger import logger
import os

# Official Gemini embeddings SDK (guaranteed output_dimensionality works)
from google import genai
from google.genai import types

router = APIRouter()


def embed_query_768(query: str, api_key: str) -> List[float]:
    """
    Returns a 768-d embedding for a query using Gemini Embedding model.
    Guaranteed to be 768 if output_dimensionality is set to 768.
    """
    client = genai.Client(api_key=api_key)
    resp = client.models.embed_content(
        model="gemini-embedding-001",
        contents=query,
        config=types.EmbedContentConfig(
            output_dimensionality=768,
            task_type="RETRIEVAL_QUERY",
        ),
    )
    return resp.embeddings[0].values


@router.post("/ask/")
async def ask_question(question: str = Form(...)):
    try:
        logger.info(f"user query: {question}")

        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medicalindex")

        if not GOOGLE_API_KEY:
            return JSONResponse(status_code=500, content={"error": "GOOGLE_API_KEY not set"})
        if not PINECONE_API_KEY:
            return JSONResponse(status_code=500, content={"error": "PINECONE_API_KEY not set"})

        # Pinecone setup
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)

        # Embed query (768-d)
        embedded_query = embed_query_768(question, GOOGLE_API_KEY)

        # Retrieve
        res = index.query(vector=embedded_query, top_k=3, include_metadata=True)

        # Convert matches -> LangChain Documents
        docs: List[Document] = []
        for match in res.get("matches", []):
            md = match.get("metadata", {}) or {}
            docs.append(
                Document(
                    page_content=md.get("text", ""),  # stored during upload
                    metadata=md
                )
            )

        # ✅ FIXED retriever (Pydantic-safe)
        class SimpleRetriever(BaseRetriever):
            docs: List[Document] = Field(default_factory=list)

            def _get_relevant_documents(self, query: str) -> List[Document]:
                return self.docs

        retriever = SimpleRetriever(docs=docs)

        # Your LLM chain
        chain = get_llm_chain(retriever)
        result = query_chain(chain, question)

        logger.info("query successful")
        return result

    except Exception as e:
        logger.exception("Error processing question")
        return JSONResponse(status_code=500, content={"error": str(e)})
