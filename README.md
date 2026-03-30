# Glyph 🔮

**An intelligent question-answering system for public-domain texts using Natural Language Processing and Retrieval-Augmented Generation**

<div align="center">

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![NLP](https://img.shields.io/badge/NLP-NLTK-orange.svg)

</div>

---

## 📖 Overview

Glyph is a desktop question-answering system that retrieves and synthesizes answers from a local library of public-domain documents. Built as a term project for AIT 626 (Natural Language Processing) at George Mason University, it demonstrates advanced NLP techniques including TF-IDF document ranking, WordNet query expansion, and RAG (Retrieval-Augmented Generation) with local LLMs.

**What makes Glyph unique:**
- **Multi-format support**: Processes `.txt`, `.html`, and `.pdf` files seamlessly
- **Intelligent two-column PDF parsing**: Automatically detects and correctly reads multi-column layouts
- **Four answer types**: LLM-generated, extractive RAG, single-sentence, and n-gram summarization
- **Smart query processing**: Anchor/signal token classification prevents common-word hijacking
- **Real-time UI**: Tkinter-based interface with live query processing

---

## 🎯 Key Features

### Advanced NLP Pipeline

1. **TF-IDF Document Ranking** - Identifies most relevant documents using cosine similarity with IDF weighting
2. **WordNet Query Expansion** - Expands query terms with synonyms (e.g., "die" → "poison", "perish", "slay")
3. **Anchor/Signal Scoring** - Classifies query tokens to prevent proper-noun repetition from dominating scores
4. **Must-Have Filtering** - Soft proper-noun filter ensures topical relevance
5. **Boilerplate Suppression** - Filters out stage directions, headers, and translator commentary

### Multiple Answer Modes

- **LLM Answer** (via Ollama): Generative responses with grounded RAG context
- **RAG Answer**: Top 3 passages stitched together in reading order
- **One-Sentence**: Single highest-scoring passage with length penalty
- **N-Gram Summary**: Blended unigram salience + query n-gram overlap

### Technical Highlights

- Clean, modular architecture with separation of concerns
- Comprehensive preprocessing pipeline (tokenization, lemmatization, stopword removal)
- Automatic logging of all queries and results
- Column-aware PDF text extraction for scanned documents
- Project Gutenberg boilerplate detection and removal

---

## 🚀 Demo

### Example Queries

```
Q: How does Romeo die?
→ Retrieves passages about Romeo's suicide with poison in the tomb

Q: What is John's profession in The Yellow Wallpaper?
→ Identifies John as a physician

Q: What does Sun Tzu say about winning without fighting?
→ Finds relevant passages from The Art of War

Q: What are the two major classes in the Communist Manifesto?
→ Extracts information about the bourgeoisie and proletariat
```

### Interface

The Tkinter GUI provides:
- Query input with real-time processing
- Document selection dropdown (search all or target specific documents)
- Ollama model selection for LLM answers
- Automatic query logging with timestamps
- Clean result display with evidence passages and scores

---

## 📦 Installation

### Prerequisites

- **Python 3.9+** (tested on Python 3.11)
- **Anaconda** (recommended for clean NLTK data management)
- **Ollama** (optional, for LLM answer layer) - [Download](https://ollama.com)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/LindseyCHill/glyph.git
   cd glyph
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download required NLTK data** (one-time setup)
   ```bash
   python -m nltk.downloader punkt stopwords wordnet omw-1.4
   ```

4. **Optional: Install and configure Ollama**
   ```bash
   # Install from https://ollama.com
   # Pull a model
   ollama pull qwen2.5:7b-instruct
   # Start the Ollama service
   ollama serve
   ```

---

## 🎮 Usage

### Running Glyph

```bash
python glyph_main.py
```

The application window will open. Documents in the `data/` folder are automatically loaded and preprocessed at startup (~5 seconds for included samples).

### Query Interface

1. Type your question in the query field
2. Select search scope: "All Documents" or a specific document
3. Choose an Ollama model (or "No model" for extractive-only answers)
4. Press Enter or click "Run"
5. View results in the output pane
6. Check `glyph_log.txt` for full session history

### Adding Your Own Documents

Simply drop `.txt`, `.html`, or `.pdf` files into the `data/` folder and click "Refresh Library" in the UI.

---

## 📊 Architecture

```
glyph_main.py              # UI entry point and query orchestration
├── glyph_document.py      # Document dataclass
├── glyph_loaders.py       # File loading (.txt, .html, .pdf)
├── glyph_preprocess.py    # Text normalization and tokenization
├── glyph_doc_scoring.py   # TF-IDF document ranking
├── glyph_summarize_ngram.py  # Extractive summarization
└── glyph_rag_ollama.py    # Ollama RAG integration
```

### Query Pipeline

```
User Query
    ↓
Query Preprocessing (tokenize, lemmatize, expand synonyms)
    ↓
Document Ranking (TF-IDF cosine similarity)
    ↓
Must-Have Filter (proper noun relevance check)
    ↓
Passage Retrieval (anchor/signal scoring)
    ↓
Boilerplate Filtering
    ↓
Answer Generation
    ├─→ LLM Answer (Ollama RAG)
    ├─→ RAG Answer (top 3 passages)
    ├─→ One-Sentence (best single passage)
    └─→ N-Gram Summary (query-focused extractive)
```

---

## 🛠️ Technologies & Skills Demonstrated

### Natural Language Processing
- **TF-IDF & Cosine Similarity**: Document and passage ranking
- **WordNet Integration**: Synonym expansion and lemmatization
- **N-gram Analysis**: Query-focused summarization
- **Named Entity Recognition**: Proper noun extraction for filtering
- **Text Preprocessing**: Tokenization, stopword removal, normalization

### Machine Learning & AI
- **Retrieval-Augmented Generation (RAG)**: Context-grounded LLM responses
- **Local LLM Integration**: Ollama subprocess interface (Lab 4 pattern)
- **Prompt Engineering**: Grounded prompts to prevent hallucination

### Python Development
- **Object-Oriented Design**: Clean separation of concerns with dataclasses
- **Type Hints**: Full type annotations for maintainability
- **Modular Architecture**: Each component has a single, clear responsibility
- **Error Handling**: Graceful degradation and user-friendly error messages

### Data Processing
- **Multi-format Parsing**: PDF (pdfplumber), HTML (BeautifulSoup), plain text
- **Column Detection**: Automatic two-column PDF layout handling
- **Boilerplate Filtering**: Project Gutenberg header/footer removal

### UI Development
- **Tkinter GUI**: Clean, responsive desktop interface
- **Async Processing**: Background document loading with progress feedback
- **Logging**: Comprehensive query history with timestamps

---

## 📚 Included Sample Documents

All sample documents are in the **public domain**:

- **Romeo and Juliet** by William Shakespeare (HTML)
- **The Yellow Wall-Paper** by Charlotte Perkins Gilman (PDF - original 1892 scan)
- **The Art of War** by Sun Tzu, translated by Lionel Giles (TXT)
- **The Communist Manifesto** by Karl Marx and Friedrich Engels (TXT)

Sources:
- Project Gutenberg: https://www.gutenberg.org
- Internet Archive: https://archive.org/details/yellowwallpaper00gilm

---

## 🧪 Example Queries & Results

### Easy Queries
```
Q: What house is Romeo from?
A: "Romeo is from the house of Montague."

Q: Who kills Mercutio?
A: "Tybalt kills Mercutio."
```

### Advanced Queries (Tests Expansion & Signal Scoring)
```
Q: How does Romeo die?
→ Correctly finds poison scene despite "die" not appearing verbatim
→ WordNet expansion: die → poison, perish, slay

Q: How does isolation affect the narrator in The Yellow Wallpaper?
→ Signal scoring prioritizes "isolation" + "affect" over character names
→ Returns psychological deterioration passages
```

---

## 🔍 Known Limitations

- **Cross-document synthesis**: Each document is queried independently; no multi-document inference
- **Inferential queries**: "Who kills X?" may retrieve nearby context but not the exact attribution sentence
- **Translator commentary**: Art of War commentaries (Chang Yu, Tu Mu) score competitively with primary text
- **PDF artifacts**: Some residual column-merge artifacts near headers in Yellow Wall-Paper scan

---

## 📝 Development Notes

**Built with AI assistance**: This project was developed with support from AI coding assistants (Claude, GitHub Copilot) for:
- Python syntax and best practices
- NLTK API usage and text processing techniques
- Debugging and optimization suggestions

**Academic context**: Created as a term project for AIT 626 Natural Language Processing, applying concepts from labs on:
- Lab 1: Text preprocessing and tokenization
- Lab 2: Extractive summarization with unigram salience
- Lab 3: Keyword-based document ranking
- Lab 4: TF-IDF retrieval and RAG with Ollama

**Solo project**: All design decisions, architecture, and integration work by Lindsey Hill.

---

## 📄 License

MIT License - See LICENSE file for details.

The included public domain texts (Romeo and Juliet, The Yellow Wall-Paper, The Art of War, The Communist Manifesto) retain their public domain status.

---

## 👤 Author

**Lindsey Hill**  
Master's in Applied Information Technology - Data Analytics & Intelligence  
George Mason University  

📧 [Email](LindseyCHill19007@gmail.com)  
💼 [LinkedIn](https://www.linkedin.com/in/lindseychill/)  
🐙 [GitHub](https://github.com/LindseyCHill)

---

## 🙏 Acknowledgments

- **NLTK Team** for the comprehensive natural language processing toolkit
- **Project Gutenberg** for the freely available literary texts
- **Ollama** for making local LLM inference accessible
- **Dr Heidari** for guidance in AIT 626 Natural Language Processing
- **AI Coding Assistants** (Claude) for development support

---

## 🗺️ Future Enhancements

- [ ] Cross-document query synthesis
- [ ] Web-based interface (Flask/FastAPI)
- [ ] Vector database integration (ChromaDB, FAISS)
- [ ] Support for additional file formats (.docx, .epub)
- [ ] Query history and favorites
- [ ] Batch query processing
- [ ] Evaluation metrics (BLEU, ROUGE, answer accuracy)

---

<div align="center">

**⭐ If you find this project interesting, consider starring it on GitHub! ⭐**

</div>
