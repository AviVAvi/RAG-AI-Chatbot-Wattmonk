# ⚡ Wattmonk AI Assistant — RAG Chatbot

An AI-powered chatbot that answers questions from multiple knowledge bases using Retrieval-Augmented Generation (RAG). Built as part of an AI internship assignment at Wattmonk.

## Features

### Core
- 🟡 **Wattmonk context** — answers questions about Wattmonk's services, technology, and company info
- 🔵 **NEC context** — answers questions about NEC electrical code and solar installation regulations  
- ⚪ **General context** — handles everyday questions using Gemini's base knowledge
- 🎯 **Confidence scoring** — shows how closely the retrieved context matched the query (green/yellow/red)
- 🧠 **Conversation memory** — maintains context across multiple exchanges in the same chat
- 💬 **Multiple chat sessions** — create, switch between, and delete separate conversations
- 🔄 **Rate limit handling** — automatically retries on API rate limits with friendly error messages

### Bonus
- 💡 **Suggested follow-up questions** — after each answer, 3 clickable follow-up questions are shown
- 🏷️ **Source attribution** — every response clearly shows which knowledge base was used

## Tech Stack
| Layer | Tool |
|---|---|
| Frontend | Streamlit |
| Backend | FastAPI |
| Vector Database | ChromaDB (local) |
| AI Model | Google Gemini 2.5 Flash |
| Embeddings | Gemini Embedding 001 |

## Project Structure
```
rag-chatbot-assignment/
├── backend/
│   └── main.py          # FastAPI app — intent classification, retrieval, Gemini calls
├── frontend/
│   └── app.py           # Streamlit chat UI with multi-chat and suggested questions
├── scripts/
│   └── ingest.py        # One-time script to load PDFs into ChromaDB
├── data/
│   └── pdfs/            # PDF knowledge base documents
├── .env.example         # Environment variable template
└── requirements.txt     # Python dependencies
```

## How RAG Works

1. **Ingestion (run once)** — PDFs are parsed → split into overlapping chunks → each chunk is converted to a vector embedding → stored in ChromaDB with a source label
2. **Query (every message)** — User asks a question → classified as wattmonk/nec/general → question is embedded → ChromaDB finds the most similar chunks → chunks injected into Gemini prompt → Gemini answers using that context
3. **Confidence Score** — Cosine similarity between the query embedding and the best retrieved chunk, shown as a percentage

## Architecture
```
User (Streamlit UI)
      ↓
FastAPI Backend (/chat endpoint)
      ↓
Intent Classifier (Gemini) → wattmonk / nec / general
      ↓
ChromaDB Retriever → top 4 most relevant chunks
      ↓
Prompt Builder → injects chunks into Gemini prompt
      ↓
Gemini 2.5 Flash → generates response + suggested questions
      ↓
Streamlit UI → displays response + badges + confidence + suggestions
```

## Setup Instructions

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd rag-chatbot-assignment
```

### 2. Create and activate virtual environment
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up your Gemini API key
```bash
cp .env.example .env
# Open .env and add your Gemini API key
```
Get a free API key at: https://aistudio.google.com

### 5. Add PDF documents to `data/pdfs/`
- `wattmonk_brochure.pdf`
- `wattmonk_info.pdf`
- `nec_guidelines.pdf`

### 6. Run the ingestion pipeline (once only)
```bash
python scripts/ingest.py
```

### 7. Start the backend (Terminal 1)
```bash
uvicorn backend.main:app --reload --port 8000
```

### 8. Start the frontend (Terminal 2)
```bash
streamlit run frontend/app.py
```

Open http://localhost:8501 in your browser.

## API Reference

### POST /chat
**Request:**
```json
{
  "message": "What services does Wattmonk offer?",
  "history": [
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello! How can I help?"}
  ]
}
```
**Response:**
```json
{
  "response": "Wattmonk offers solar sales proposals...",
  "source": "wattmonk",
  "context_used": true,
  "confidence": 79.8,
  "suggested_questions": [
    "How fast does Wattmonk deliver plansets?",
    "What is Zippy?",
    "How many states does Wattmonk cover?"
  ]
}
```

### GET /health
Returns `{"status": "ok"}` if the backend is running.