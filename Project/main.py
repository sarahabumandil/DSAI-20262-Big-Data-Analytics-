# main.py
#
# FastAPI backend for the Multimodal Financial RAG app, designed to run as a
# single Vercel Serverless Function (see /vercel.json). It is fully
# stateless: every request opens a short-lived connection to Weaviate Cloud,
# does its work, and closes it again. No state is kept in memory or on disk
# between invocations, and no local ML models are loaded — all embedding,
# vision, and generation calls go to Weaviate Cloud / Groq Cloud.
#
# Required environment variables (set these in the Vercel project settings):
#   GROQ_API_KEY        - https://console.groq.com
#   WEAVIATE_URL        - your Weaviate Cloud (WCS) cluster REST endpoint
#   WEAVIATE_API_KEY    - your Weaviate Cloud API key
#
# The Weaviate collection is configured to use Weaviate's own hosted
# vectorizer (text2vec-weaviate), so no separate embeddings API key is
# needed. If your cluster doesn't have that module enabled, swap the
# vectorizer_config in ensure_collection() for text2vec-cohere / -openai
# and add the matching header/API key.

import os
import uuid
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import Filter

from document_processors import load_multimodal_data
from utils import chunk_text, chat_completion

COLLECTION_NAME = "FinancialRagChunk"

app = FastAPI(title="Multimodal Financial RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Weaviate connection helpers
# ---------------------------------------------------------------------------

def get_weaviate_client():
    url = os.environ.get("WEAVIATE_URL")
    api_key = os.environ.get("WEAVIATE_API_KEY")
    if not url or not api_key:
        raise HTTPException(
            status_code=500,
            detail="WEAVIATE_URL / WEAVIATE_API_KEY are not configured on the server.",
        )
    return weaviate.connect_to_weaviate_cloud(
        cluster_url=url,
        auth_credentials=Auth.api_key(api_key),
    )


def ensure_collection(client):
    if not client.collections.exists(COLLECTION_NAME):
        client.collections.create(
            name=COLLECTION_NAME,
            vectorizer_config=Configure.Vectorizer.text2vec_weaviate(),
            properties=[
                Property(name="text", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
                Property(name="doc_type", data_type=DataType.TEXT),
                Property(name="page_num", data_type=DataType.INT),
                Property(name="session_id", data_type=DataType.TEXT),
            ],
        )
    return client.collections.get(COLLECTION_NAME)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    session_id: str
    query: str
    top_k: Optional[int] = 5


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]


class UploadResponse(BaseModel):
    session_id: str
    files_processed: int
    documents_extracted: int
    chunks_indexed: int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/upload", response_model=UploadResponse)
async def upload_files(
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = Form(None),
):
    """Accept one or more files, extract + describe their content in-memory,
    chunk it, and index it into Weaviate under a session_id so concurrent
    users never see each other's documents."""
    session_id = session_id or str(uuid.uuid4())

    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")

    file_payloads = []
    for f in files:
        content = await f.read()
        file_payloads.append((f.filename, content))

    try:
        raw_documents = load_multimodal_data(file_payloads)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document processing failed: {e}")

    chunk_count = 0
    client = get_weaviate_client()
    try:
        collection = ensure_collection(client)
        with collection.batch.dynamic() as batch:
            for doc in raw_documents:
                metadata = doc.get("metadata", {})
                for chunk in chunk_text(doc.get("text", "")):
                    batch.add_object(properties={
                        "text": chunk,
                        "source": metadata.get("source", "unknown"),
                        "doc_type": metadata.get("type", "text"),
                        "page_num": int(metadata.get("page_num") or 0),
                        "session_id": session_id,
                    })
                    chunk_count += 1
    finally:
        client.close()

    return UploadResponse(
        session_id=session_id,
        files_processed=len(files),
        documents_extracted=len(raw_documents),
        chunks_indexed=chunk_count,
    )


@app.post("/api/query", response_model=QueryResponse)
def query_documents(payload: QueryRequest):
    """Retrieve the most relevant chunks for this session and ask the LLM
    to answer the question grounded in them."""
    client = get_weaviate_client()
    try:
        if not client.collections.exists(COLLECTION_NAME):
            raise HTTPException(
                status_code=404,
                detail="No documents have been indexed yet. Upload files first.",
            )
        collection = client.collections.get(COLLECTION_NAME)
        results = collection.query.near_text(
            query=payload.query,
            filters=Filter.by_property("session_id").equal(payload.session_id),
            limit=payload.top_k or 5,
        )
    finally:
        client.close()

    if not results.objects:
        return QueryResponse(
            answer=(
                "I couldn't find anything relevant in the documents uploaded "
                "for this session. Try uploading files first, or rephrase "
                "your question."
            ),
            sources=[],
        )

    context_parts = []
    sources = []
    for obj in results.objects:
        props = obj.properties
        context_parts.append(f"[Source: {props.get('source')}] {props.get('text')}")
        sources.append({
            "source": props.get("source"),
            "type": props.get("doc_type"),
            "page_num": props.get("page_num"),
        })

    context = "\n\n".join(context_parts)
    system_prompt = (
        "You are a meticulous financial research assistant. Answer the "
        "user's question using ONLY the provided context extracted from "
        "their uploaded financial documents. If the context doesn't contain "
        "enough information, say so plainly rather than guessing. Reference "
        "source names inline where it's useful."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {payload.query}"

    try:
        answer = chat_completion(system_prompt, user_prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {e}")

    return QueryResponse(answer=answer, sources=sources)


@app.delete("/api/session/{session_id}")
def clear_session(session_id: str):
    """Delete all indexed chunks belonging to a session (e.g. on 'Clear Chat')."""
    client = get_weaviate_client()
    try:
        if client.collections.exists(COLLECTION_NAME):
            collection = client.collections.get(COLLECTION_NAME)
            collection.data.delete_many(
                where=Filter.by_property("session_id").equal(session_id)
            )
    finally:
        client.close()
    return {"status": "cleared", "session_id": session_id}
