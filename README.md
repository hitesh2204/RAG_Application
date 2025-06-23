# 🤖 Multimodal RAG: PDF + YouTube + Image + Text QA System

This is a **Retrieval-Augmented Generation (RAG)** application that answers questions using information from **text files**, **PDFs**, **images (OCR)**, and **YouTube videos** (transcripts). It integrates **LangChain**, **FAISS**, **HuggingFace Transformers**, and **Tesseract OCR** to build a unified multimodal knowledge system.
---
## 🔍 Key Features

✅ OCR from images using Tesseract  
✅ PDF parsing with `PyPDFLoader`  
✅ Text file loading  
✅ YouTube video transcript extraction  
✅ Chunking, embedding, and vector search with FAISS  
✅ Query answering via LLM (Flan-T5)  
✅ Fully local code (except for HuggingFace API)

---

## 📁 Folder Structure
Multimodal_RAG/
├── data/
│ ├── news.txt # Plain text file
│ ├── ML cheetsheet.pdf # PDF document
│ └── rcb.jpeg # Image for OCR
├── main.py # Main Python script
└── README.md # This documentation


---

## 🧰 Tech Stack

| Component         | Tool Used                                      |
|------------------|------------------------------------------------|
| OCR              | Tesseract OCR                                  |
| PDF Parsing      | LangChain PyPDFLoader                          |
| Video Transcript | LangChain YoutubeLoader                        |
| Embeddings       | `all-MiniLM-L6-v2` via HuggingFace Transformers|
| Vector Store     | FAISS                                           |
| Language Model   | `google/flan-t5-large` (HuggingFace Hub)       |
| Chain Framework  | LangChain                                      |

---

| Component         | Tool Used                                      |
|------------------|------------------------------------------------|
| OCR              | Tesseract OCR                                  |
| PDF Parsing      | LangChain PyPDFLoader                          |
| Video Transcript | LangChain YoutubeLoader                        |
| Embeddings       | `all-MiniLM-L6-v2` via HuggingFace Transformers|
| Vector Store     | FAISS                                           |
| Language Model   | `google/flan-t5-large` (HuggingFace Hub)       |
| Chain Framework  | LangChain                                      |

## ⚙️ Installation & Setup

### 1. Clone the Repo

```bash
git clone https://github.com/your-username/multimodal-rag.git
cd multimodal-rag

2. Install Python Dependencies
bash
Copy
Edit
pip install -r requirements.txt
pillow
pytesseract
opencv-python
langchain
faiss-cpu
sentence-transformers
huggingface_hub
youtube-transcript-api

3. Install Tesseract OCR
Download and install from:
👉 https://github.com/tesseract-ocr/tesseract

Update the path in main.py if you're on Windows:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

4. HuggingFace API Key
Set your HuggingFace Hub token:
export HUGGINGFACEHUB_API_TOKEN=your_token_here

🚀 Run the Application
python main.py

🔄 Loading all data sources...
🧱 Splitting documents into chunks...
🔍 Building vector store...
🧠 Loading LLM...
💬 Answering query...

📜 Final Answer:
<Your final answer based on retrieved docs>

🧪 Example Query
python
Copy
user_query = "What did the video talk about?"
The system will extract transcript from YouTube, OCR from image, PDF content, and plain text, and then use an LLM to generate an answer.

| Source      | Method                        |
| ----------- | ----------------------------- |
| Text file   | `TextLoader`                  |
| PDF file    | `PyPDFLoader`                 |
| Image file  | `pytesseract.image_to_string` |
| YouTube URL | `YoutubeLoader` (transcripts) |

🙋 Author
Built by Hitesh Yerekar
🔬 Exploring Multimodal AI | GenAI | LLMs | LangChain | RAG

