import os
import time
import streamlit as st
import chromadb
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Wattmonk AI Assistant",
    page_icon="⚡",
    layout="centered"
)

# --- Initialize Gemini client ---
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# --- Initialize ChromaDB ---
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "../data/chroma_db")

@st.cache_resource
def get_collection():
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    return chroma_client.get_or_create_collection(
        name="knowledge_base",
        metadata={"hnsw:space": "cosine"}
    )

collection = get_collection()

st.markdown("""
<style>
.source-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 500;
    margin-bottom: 4px;
}
.badge-wattmonk { background-color: #FFF3CD; color: #856404; }
.badge-nec      { background-color: #D1ECF1; color: #0C5460; }
.badge-general  { background-color: #E2E3E5; color: #383D41; }
</style>
""", unsafe_allow_html=True)

# --- Session state setup ---
if "chats" not in st.session_state:
    st.session_state.chats = {"Chat 1": []}
if "current_chat" not in st.session_state:
    st.session_state.current_chat = "Chat 1"
if "chat_counter" not in st.session_state:
    st.session_state.chat_counter = 1
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# --- Helper functions ---
def create_new_chat():
    st.session_state.chat_counter += 1
    chat_name = f"Chat {st.session_state.chat_counter}"
    st.session_state.chats[chat_name] = []
    st.session_state.current_chat = chat_name

def delete_chat(chat_name):
    if len(st.session_state.chats) == 1:
        st.session_state.chats[chat_name] = []
        return
    del st.session_state.chats[chat_name]
    st.session_state.current_chat = list(st.session_state.chats.keys())[-1]

def get_chat_preview(messages):
    for msg in messages:
        if msg["role"] == "user":
            preview = msg["content"][:25]
            return preview + "..." if len(msg["content"]) > 25 else preview
    return "New conversation"

def render_badges(source, confidence):
    badge_class = f"badge-{source}"
    label = {
        "wattmonk": "🟡 Wattmonk docs",
        "nec": "🔵 NEC code",
        "general": "⚪ General"
    }.get(source, source)

    if confidence and confidence > 0:
        confidence_color = (
            "#28a745" if confidence >= 70
            else "#ffc107" if confidence >= 40
            else "#dc3545"
        )
        st.markdown(
            f'<span class="source-badge {badge_class}">{label}</span> '
            f'<span style="background-color:{confidence_color};color:white;'
            f'padding:2px 10px;border-radius:12px;font-size:12px;font-weight:500;">'
            f'🎯 {confidence}% match</span>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<span class="source-badge {badge_class}">{label}</span>',
            unsafe_allow_html=True
        )

# --- RAG functions ---
def gemini_generate_with_retry(prompt, max_retries=3):
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
                    time.sleep(wait_time)
                else:
                    raise Exception("Rate limit reached. Please wait a moment and try again.")
            else:
                raise e

def classify_intent(message):
    prompt = """You are a query classifier for a solar industry chatbot.

Classify the user's query into exactly one of these categories:
- "wattmonk": questions about Wattmonk company, its services, pricing, team, technology, or any of its products like Zippy, plansets, PTO applications, or solar permits
- "nec": questions about NEC electrical code, wiring standards, or solar installation regulations
- "general": everything else

Respond with ONLY one word: wattmonk, nec, or general.

User query: """ + message

    result = gemini_generate_with_retry(prompt)
    if result.lower() not in ["wattmonk", "nec", "general"]:
        return "general"
    return result.lower()

def get_embedding(text, task_type="RETRIEVAL_QUERY"):
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=types.EmbedContentConfig(task_type=task_type)
    )
    return result.embeddings[0].values

def retrieve_context(query, source_filter, top_k=4):
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

Provide a clear, detailed answer based on the context above."""
    return prompt

def generate_suggested_questions(message, answer, intent):
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

def process_message(message, history):
    intent = classify_intent(message)
    context_chunks = []
    context_used = False

    if intent in ["wattmonk", "nec"]:
        context_chunks = retrieve_context(message, source_filter=intent)
        context_used = len(context_chunks) > 0

    final_message = build_prompt(message, context_chunks, intent)

    history_for_gemini = []
    for msg in history[-6:]:
        role = "user" if msg["role"] == "user" else "model"
        history_for_gemini.append({
            "role": role,
            "parts": [{"text": msg["content"]}]
        })

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
        else:
            answer = f"Something went wrong: {str(e)}"

    confidence = (
        round(context_chunks[0]["similarity"] * 100, 1)
        if context_chunks else 0.0
    )

    suggested_questions = generate_suggested_questions(message, answer, intent)

    return {
        "response": answer,
        "source": intent,
        "context_used": context_used,
        "confidence": confidence,
        "suggested_questions": suggested_questions
    }

def send_message(prompt):
    current_messages = st.session_state.chats[st.session_state.current_chat]
    current_messages.append({"role": "user", "content": prompt})

    history = current_messages[:-1]

    with st.spinner("Thinking..."):
        try:
            data = process_message(prompt, history)

            current_messages.append({
                "role": "assistant",
                "content": data["response"],
                "source": data["source"],
                "context_used": data["context_used"],
                "confidence": data["confidence"],
                "suggested_questions": data["suggested_questions"]
            })
            st.session_state.chats[st.session_state.current_chat] = current_messages

        except Exception as e:
            st.error(f"Something went wrong: {str(e)}")

# --- Sidebar ---
with st.sidebar:
    st.header("About")
    st.markdown("""
    This chatbot can answer questions from:

    🟡 **Wattmonk** — company info, services, technology

    🔵 **NEC Code** — electrical standards and regulations

    ⚪ **General** — anything else
    """)

    st.divider()

    if st.button("➕ New Chat", use_container_width=True):
        create_new_chat()
        st.rerun()

    st.caption("Your conversations")

    for chat_name in list(st.session_state.chats.keys()):
        messages = st.session_state.chats[chat_name]
        preview = get_chat_preview(messages)
        is_active = chat_name == st.session_state.current_chat

        col1, col2 = st.columns([5, 1])
        with col1:
            if st.button(
                f"💬 {preview}",
                key=f"chat_{chat_name}",
                use_container_width=True,
                type="primary" if is_active else "secondary"
            ):
                st.session_state.current_chat = chat_name
                st.rerun()
        with col2:
            if st.button("🗑", key=f"del_{chat_name}"):
                delete_chat(chat_name)
                st.rerun()

    st.divider()
    st.caption("Powered by Google Gemini + ChromaDB RAG")

# --- Main area ---
st.title("⚡ Wattmonk AI Assistant")
st.caption("Ask me about Wattmonk's services, NEC electrical codes, or anything solar-related.")

if st.session_state.pending_question:
    pending = st.session_state.pending_question
    st.session_state.pending_question = None
    send_message(pending)
    st.rerun()

current_messages = st.session_state.chats[st.session_state.current_chat]

for i, msg in enumerate(current_messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and "source" in msg:
            render_badges(msg["source"], msg.get("confidence", 0.0))
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("suggested_questions"):
            st.markdown("**💡 You might also want to ask:**")
            for j, q in enumerate(msg["suggested_questions"]):
                if st.button(q, key=f"sq_{i}_{j}"):
                    st.session_state.pending_question = q
                    st.rerun()

if prompt := st.chat_input("Ask a question..."):
    send_message(prompt)
    st.rerun()
