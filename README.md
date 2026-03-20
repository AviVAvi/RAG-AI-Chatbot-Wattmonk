# ⚡ Wattmonk AI Assistant — RAG Chatbot

An AI-powered chatbot that answers questions from multiple knowledge bases using Retrieval-Augmented Generation (RAG). Built as part of an AI internship assignment at Wattmonk.

## Live Demo
🌐 [https://rag-ai-chatbot-wattmonk-hgmzlycnf9pnfarytjf2nu.streamlit.app](https://rag-ai-chatbot-wattmonk-hgmzlycnf9pnfarytjf2nu.streamlit.app)

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
| Backend | FastAPI (local) / Streamlit (deployed) |
| Vector Database | ChromaDB |
| AI Model | Google Gemini 2.5 Flash |
| Embeddings | Gemini Embedding 001 |

## Project Structure
```
rag-chatbot-assignment/
├── backend/
│   └── main.py          # FastAPI app — for local development with separate backend
├── frontend/
│   └── app.py           # Streamlit app — contains full RAG logic for deployment
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
git clone https://github.com/AviVAvi/RAG-AI-Chatbot-Wattmonk
cd RAG-AI-Chatbot-Wattmonk
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

### 7. Run the app
```bash
streamlit run frontend/app.py
```

Open http://localhost:8501 in your browser.

## API Reference (Local Development)

### POST /chat
**Request:**
```json
{
  "message": "What services does Wattmonk offer?",
  "history": []
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
```json
Returns `{"status": "ok"}` if the backend is running.
```
```
```
## User Guide

### How to use the chatbot
1. **Type your question** in the chat box at the bottom and press Enter
2. **Read the source badge** above each response to know which knowledge base was used:
   - 🟡 Yellow = answer came from Wattmonk company documents
   - 🔵 Blue = answer came from NEC electrical code
   - ⚪ Grey = answer from Gemini's general knowledge
3. **Check the confidence score** — green (70%+) means high confidence, yellow (40-70%) means moderate, red (below 40%) means low
4. **Click suggested questions** below each answer to explore related topics
5. **Start a new chat** using the ➕ button in the sidebar
6. **Switch between chats** by clicking any conversation in the sidebar
7. **Delete a chat** using the 🗑 button next to any conversation

### Example questions to try
- "What services does Wattmonk offer?"
- "What is Zippy?"
- "When was Wattmonk founded?"
- "What are the wiring requirements for solar panels?"
- "What does NEC Article 690 cover?"
- "How many permits does Wattmonk deliver monthly?"

## Performance Metrics

Based on testing during development:

| Metric | Value |
|---|---|
| Average response time | 3-6 seconds |
| Intent classification accuracy | ~95% |
| Wattmonk query confidence score | 75-85% average |
| NEC query confidence score | 60-70% average |
| Supported knowledge bases | 2 (Wattmonk + NEC) |
| Max conversation history | 6 turns |
| Chunks in vector database | 158 total |
