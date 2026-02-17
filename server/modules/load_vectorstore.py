import os
import time
from pathlib import Path
from dotenv import load_dotenv
from tqdm.auto import tqdm

from pinecone import Pinecone, ServerlessSpec

# Old LangChain syntax (as you asked)
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Official Gemini embeddings SDK (guaranteed output_dimensionality works)
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Pinecone config
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")   # region
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medicalindex")

UPLOAD_DIR = "./uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY not set")
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY not set")

# Initialize Pinecone instance
pc = Pinecone(api_key=PINECONE_API_KEY)
spec = ServerlessSpec(cloud="aws", region=PINECONE_ENV)

# Create index if not exists (768)
existing_indexes = [i["name"] for i in pc.list_indexes()]
if PINECONE_INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=768,
        metric="dotproduct",
        spec=spec
    )
    while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
        time.sleep(1)

index = pc.Index(PINECONE_INDEX_NAME)

# Gemini client (reuse)
client = genai.Client(api_key=GOOGLE_API_KEY)

def embed_texts_768(texts):
    """
    Batch embeds a list of strings -> list[list[float]] each 768-d.
    """
    resp = client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts,
        config=types.EmbedContentConfig(
            output_dimensionality=768,
            task_type="RETRIEVAL_DOCUMENT",
        ),
    )
    return [e.values for e in resp.embeddings]

def load_vectorstore(uploaded_files):
    file_paths = []

    # 1) Save uploaded files
    for file in uploaded_files:
        save_path = Path(UPLOAD_DIR) / file.filename
        with open(save_path, "wb") as f:
            f.write(file.file.read())
        file_paths.append(save_path)

    # 2) Process each file
    for file_path in file_paths:
        loader = PyPDFLoader(str(file_path))
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
        )
        chunks = splitter.split_documents(documents)

        texts = [chunk.page_content for chunk in chunks]
        # IMPORTANT: store text inside metadata, so ask_question can read metadata["text"]
        metadatas = [{**chunk.metadata, "text": chunk.page_content} for chunk in chunks]
        ids = [f"{Path(file_path).stem}-{i}" for i in range(len(chunks))]

        # 3) Embedding (guaranteed 768)
        print(f"🔍 Embedding {len(texts)} chunks (768-d)...")
        embeddings = embed_texts_768(texts)

        # Safety check
        if embeddings and len(embeddings[0]) != 768:
            raise RuntimeError(f"Embedding dim is {len(embeddings[0])}, expected 768")

        # 4) Upsert
        print("📤 Uploading to Pinecone...")
        with tqdm(total=len(embeddings), desc="Upserting to Pinecone") as progress:
            index.upsert(vectors=list(zip(ids, embeddings, metadatas)))
            progress.update(len(embeddings))

        print(f"✅ Upload complete for {file_path}")
