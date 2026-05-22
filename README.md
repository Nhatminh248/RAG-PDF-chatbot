# Ultra-RAG PDF Chatbot

A high-performance, premium-designed RAG (Retrieval-Augmented Generation) chatbot application that lets you upload multiple PDF documents and chat with their collective content in real-time. 

Built using a modern technical stack featuring **Streamlit**, **LangChain**, **ChromaDB**, and **OpenRouter (Gemini 2.0 Flash)**, with **HuggingFace sentence-transformers** running completely locally on your machine.

---

## ⚡ Key Features

- 🎨 **Premium Aesthetic**: Responsive, gorgeous dark mode design with neon gradients and glassmorphism elements.
- 📁 **Multi-Document Ingestion**: Upload multiple PDF files simultaneously via a drag-and-drop area.
- 📂 **Interactive Document Hub**: View and manage indexed documents in the sidebar. Remove individual documents dynamically or wipe the database clean.
- 💬 **State-of-the-Art Conversational RAG**: Engage in natural conversations with memory. Follow-up queries correctly resolve across all indexed documents.
- 🔍 **Granular Citations**: Under each AI response, view collapsible reference panels with exact text snippets, source file names, and page numbers.
- 💾 **Persistent DB & Sync**: Built on ChromaDB. The app detects existing indices on startup, preserving document management even after restarts.

---

## 🛠️ Technical Stack

- **Frontend UI**: [Streamlit](https://streamlit.io/)
- **Orchestration**: [LangChain](https://www.langchain.com/)
- **Local Embeddings**: HuggingFace [sentence-transformers](https://huggingface.co/sentence-transformers) (`all-MiniLM-L6-v2` ~90MB download)
- **Vector Database**: [ChromaDB](https://www.trychroma.com/) (Persisted locally in `./chroma_db`)
- **PDF Parser**: [pypdf](https://pypi.org/project/pypdf/)
- **LLM Provider**: OpenRouter API (`google/gemini-2.0-flash-exp:free`) via LangChain OpenAI bindings

---

## 🚀 Setup Instructions

Follow these quick steps to get the app running on your machine:

### 1. Install Dependencies
Ensure you have Python 3.9+ installed. Run the following command to install all pinned dependencies:
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy the template environment file to `.env`:
```bash
cp .env.example .env
```
Open `.env` in a text editor and replace the placeholder with your OpenRouter API key:
```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
```
*(Alternatively: You can launch the app and paste the API key directly into the secure sidebar input field!)*

### 3. Start the Application
Run the Streamlit server:
```bash
streamlit run app.py
```

---

## 💡 Important Notes

> [!NOTE]
> **Local Embeddings Run 100% Locally:**
> Embeddings are processed using the HuggingFace `all-MiniLM-L6-v2` model locally on your CPU/GPU. No API keys are required for embeddings. On the very first run, a ~90MB model file will download automatically. Subsequent runs will use the cached local file instantly.

> [!TIP]
> **ChromaDB Persistence:**
> The vector database is stored in the `./chroma_db` directory within the project folder. If you restart the app, your indexed documents list will sync automatically with whatever remains in the database.
