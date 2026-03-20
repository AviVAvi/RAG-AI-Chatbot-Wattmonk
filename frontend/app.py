import streamlit as st
import requests

st.set_page_config(
    page_title="Wattmonk AI Assistant",
    page_icon="⚡",
    layout="centered"
)

BACKEND_URL = "http://localhost:8000"

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

def send_message(prompt):
    """Send a message to the backend and display the response."""
    current_messages = st.session_state.chats[st.session_state.current_chat]

    current_messages.append({"role": "user", "content": prompt})

    history_for_api = [
        {"role": m["role"], "content": m["content"]}
        for m in current_messages[:-1][-6:]
    ]

    with st.spinner("Thinking..."):
        try:
            response = requests.post(
                f"{BACKEND_URL}/chat",
                json={
                    "message": prompt,
                    "history": history_for_api
                },
                timeout=60
            )
            response.raise_for_status()
            data = response.json()

            answer = data["response"]
            source = data["source"]
            context_used = data["context_used"]
            confidence = data.get("confidence", 0.0)
            suggested_questions = data.get("suggested_questions", [])

            current_messages.append({
                "role": "assistant",
                "content": answer,
                "source": source,
                "context_used": context_used,
                "confidence": confidence,
                "suggested_questions": suggested_questions
            })

            st.session_state.chats[st.session_state.current_chat] = current_messages

        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to backend. Make sure FastAPI is running on port 8000.")
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

# Handle pending question from suggested buttons
if st.session_state.pending_question:
    pending = st.session_state.pending_question
    st.session_state.pending_question = None
    send_message(pending)
    st.rerun()

current_messages = st.session_state.chats[st.session_state.current_chat]

# Display conversation history
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

# Chat input
if prompt := st.chat_input("Ask a question..."):
    send_message(prompt)
    st.rerun()