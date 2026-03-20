import os
import time
import chromadb
from google import genai
from google.genai import types
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

PDFS_DIR = os.path.join(os.path.dirname(__file__), "../data/pdfs")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "../data/chroma_db")
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
MAX_CHUNKS_PER_PDF = 150

SOURCE_MAP = {
    "wattmonk": ["wattmonk_brochure.pdf", "wattmonk_info.pdf"],
    "nec": ["nec_guidelines.pdf"],
}

def get_source_label(filename):
    for source, files in SOURCE_MAP.items():
        if filename in files:
            return source
    return "general"

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def get_embedding(text):
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
    )
    return result.embeddings[0].values

def ingest():
    print("Starting ingestion pipeline...")

    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Delete old collection so we start fresh
    try:
        chroma_client.delete_collection("knowledge_base")
        print("Cleared old database.")
    except:
        pass

    collection = chroma_client.get_or_create_collection(
        name="knowledge_base",
        metadata={"hnsw:space": "cosine"}
    )

    pdf_files = [f for f in os.listdir(PDFS_DIR) if f.endswith(".pdf")]

    if not pdf_files:
        print("No PDFs found in data/pdfs/.")
        return

    total_chunks = 0
    request_count = 0

    for pdf_file in pdf_files:
        pdf_path = os.path.join(PDFS_DIR, pdf_file)
        source_label = get_source_label(pdf_file)

        print(f"\nProcessing: {pdf_file} (source: {source_label})")

        text = extract_text_from_pdf(pdf_path)
        print(f"  Extracted {len(text)} characters")

        chunks = chunk_text(text)

        # Limit chunks per PDF to avoid rate limits
        if len(chunks) > MAX_CHUNKS_PER_PDF:
            print(f"  Limiting to {MAX_CHUNKS_PER_PDF} chunks (was {len(chunks)})")
            chunks = chunks[:MAX_CHUNKS_PER_PDF]

        print(f"  Processing {len(chunks)} chunks...")

        for i, chunk in enumerate(chunks):
            chunk_id = f"{source_label}_{pdf_file}_{i}"

            existing = collection.get(ids=[chunk_id])
            if existing["ids"]:
                print(f"  Chunk {i+1} already exists, skipping...")
                continue

            # Rate limit: wait 0.7 seconds between requests
            # This keeps us under 100 requests per minute
            if request_count > 0:
                time.sleep(0.7)

            embedding = get_embedding(chunk)

            collection.add(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{
                    "source": source_label,
                    "filename": pdf_file,
                    "chunk_index": i
                }]
            )

            total_chunks += 1
            request_count += 1
            print(f"  Embedded chunk {i+1}/{len(chunks)}")

        print(f"  Done — {len(chunks)} chunks stored for {pdf_file}")

    print(f"\nIngestion complete! {total_chunks} total chunks stored.")

if __name__ == "__main__":
    ingest()
