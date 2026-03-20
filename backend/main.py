import os
import time
import chromadb
from google import genai
from google.genai import types
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI(title="Wattmonk RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "../data/chroma_db")
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_or_create_collection(
    name="knowledge_base",
    metadata={"hnsw:space": "cosine"}
)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: list[Message] = []

class ChatResponse(BaseModel):
    response: str
    source: str
    context_used: bool
    confidence: float = 0.0
    suggested_questions: list[str] = []

def gemini_generate_with_retry(prompt, max_retries=3):
    """
    Call Gemini with automatic retry on rate limit errors.
    
    How it works:
    - If we hit a 429 (rate limit), wait a few seconds and try again
    - We try up to max_retries times before giving up
    - Each retry waits a bit longer than the previous one (exponential backoff)
    """
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 3
                    print(f"Rate limit hit. Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}...")
                    time.sleep(wait_time)
                else:
                    raise Exception("Rate limit reached. Please wait a moment and try again.")
            else:
                raise e

def classify_intent(message):
    """
    Determine which knowledge base to search.
    Returns: wattmonk, nec, or general
    """
    prompt = """You are a query classifier for a solar industry chatbot.

Classify the user's query into exactly one of these categories:
- - "wattmonk": questions about Wattmonk company, its services, pricing, team, technology, or any of its products like Zippy, plansets, PTO applications, or solar permits
- "nec": questions about NEC electrical code, wiring standards, or solar installation regulations
- "general": everything else

Respond with ONLY one word: wattmonk, nec, or general.

User query: """ + message

    result = gemini_generate_with_retry(prompt)

    if result.lower() not in ["wattmonk", "nec", "general"]:
        return "general"

    return result.lower()

def get_embedding(text):
    """
    Convert text to a vector embedding using Gemini.
    Used for both ingestion and query time.
    """
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    )
    return result.embeddings[0].values

def retrieve_context(query, source_filter, top_k=4):
    """
    Search ChromaDB for the most relevant chunks to the query.
    Only searches within the specified source (wattmonk or nec).
    """
    query_embedding = get_embedding(query)

    where_filter = {"source": source_filter}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_filter,
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    if results["documents"] and results["documents"][0]:
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            chunks.append({
                "text": doc,
                "metadata": meta,
                "similarity": 1 - dist
            })

    return chunks

def build_prompt(message, context_chunks, intent):
    """
    Build the final prompt with retrieved context injected.
    This is the core of RAG — giving the LLM the relevant document excerpts.
    """
    if not context_chunks:
        return message

    context_text = "\n\n---\n\n".join([chunk["text"] for chunk in context_chunks])

    source_label = (
        "Wattmonk company documentation"
        if intent == "wattmonk"
        else "NEC electrical code guidelines"
    )

    prompt = f"""You are a helpful assistant for Wattmonk, a solar engineering company.

Use the following excerpts from {source_label} to answer the user's question accurately.
Important instructions:
- Actually explain and summarize the relevant information from the context
- Do NOT just cite page numbers or section references — give the actual content
- If the context contains specific requirements, rules, or details, state them clearly
- If the answer is not in the provided context, say so honestly rather than guessing
- Keep your answer concise but informative

CONTEXT:
{context_text}

USER QUESTION:
{message}

Provide a clear, detailed answer based on the context above. Explain the actual content, not just where to find it."""

    return prompt

def generate_suggested_questions(message, answer, intent):
    """
    Generate 3 follow-up questions based on the conversation so far.
    These help the user explore the topic further.
    """
    prompt = f"""Based on this question and answer, generate exactly 3 short follow-up questions the user might want to ask next.

User asked: {message}
Answer given: {answer[:300]}
Topic area: {intent}

Rules:
- Each question must be short (max 10 words)
- Make them genuinely useful and related to the topic
- Return ONLY the 3 questions, one per line, no numbering, no bullets, no extra text

3 follow-up questions:"""

    try:
        result = gemini_generate_with_retry(prompt)
        questions = [q.strip() for q in result.strip().split("\n") if q.strip()]
        return questions[:3]
    except:
        return []

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint.
    1. Classify intent
    2. Retrieve relevant chunks from ChromaDB
    3. Build prompt with context
    4. Call Gemini and return response
    """
    message = request.message.strip()
    history = request.history

    # Step 1: Classify intent
    intent = classify_intent(message)
    context_chunks = []
    context_used = False

    # Step 2: Retrieve context if domain specific
    if intent in ["wattmonk", "nec"]:
        context_chunks = retrieve_context(message, source_filter=intent)
        context_used = len(context_chunks) > 0

    # Step 3: Build prompt
    final_message = build_prompt(message, context_chunks, intent)

    # Step 4: Build conversation history for Gemini
    history_for_gemini = []
    for msg in history[-6:]:
        role = "user" if msg.role == "user" else "model"
        history_for_gemini.append({
            "role": role,
            "parts": [{"text": msg.content}]
        })

    # Step 5: Call Gemini with retry logic
    system_context = """You are a knowledgeable assistant for Wattmonk, a solar engineering company.
You help users with questions about Wattmonk's services and NEC electrical codes.
Be concise, accurate, and professional.
For general questions, answer from your own knowledge."""

    full_message = system_context + "\n\n" + final_message

    try:
        chat_session = client.chats.create(
            model="gemini-2.5-flash",
            history=history_for_gemini
        )
        response = chat_session.send_message(full_message)
        answer = response.text.strip()
    except Exception as e:
        error_str = str(e)
        if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
            answer = "I'm currently experiencing high demand. Please wait a moment and try again."
        elif "Rate limit reached" in error_str:
            answer = "Rate limit reached. Please wait about 30 seconds and try again."
        else:
            raise e

    # Step 6: Calculate confidence score
    confidence = (
        round(context_chunks[0]["similarity"] * 100, 1)
        if context_chunks else 0.0
    )

    suggested_questions = generate_suggested_questions(message, answer, intent)

    return ChatResponse(
        response=answer,
        source=intent,
        context_used=context_used,
        confidence=confidence,
        suggested_questions=suggested_questions
    )

@app.get("/health")
def health():
    return {"status": "ok"}