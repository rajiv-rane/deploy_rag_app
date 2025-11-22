# 🏥 Clinical Summary RAG Application

An AI-powered medical documentation system that generates comprehensive discharge summaries using Retrieval-Augmented Generation (RAG) architecture. This application leverages advanced language models, medical domain embeddings, and vector databases to assist healthcare professionals in creating accurate, structured clinical documentation.

## 📋 Project Description

The Clinical Summary RAG Application is a sophisticated healthcare documentation tool designed to streamline the process of generating medical discharge summaries. The system combines:

- **RAG Architecture**: Retrieval-Augmented Generation for context-aware responses
- **Medical Domain Embeddings**: Bio ClinicalBERT for medical text understanding
- **Fast LLM Inference**: Groq API with LLaMA 4 Maverick model for rapid generation
- **Vector Search**: ChromaDB for semantic similarity search across patient records
- **Modern Web Interface**: Streamlit frontend with FastAPI backend for optimal performance

The application enables healthcare professionals to:
- Search and retrieve patient information from MongoDB
- Generate structured discharge summaries automatically
- Find similar patient cases using semantic search
- Interact with an AI assistant for clinical queries
- Export summaries in multiple formats (TXT, DOCX, PDF)
- Customize summary templates based on insurance requirements

## 🎯 Use Cases

### 1. **Automated Discharge Summary Generation**
   - **Problem**: Manual creation of discharge summaries is time-consuming and prone to inconsistencies
   - **Solution**: Automatically generate comprehensive, structured discharge summaries from patient data
   - **Benefit**: Reduces documentation time by 60-80%, ensures consistency, and minimizes errors

### 2. **Clinical Decision Support**
   - **Problem**: Healthcare providers need quick access to similar cases for reference
   - **Solution**: Semantic search across historical patient records to find similar cases
   - **Benefit**: Provides evidence-based references for treatment planning and decision-making

### 3. **AI-Powered Clinical Assistant**
   - **Problem**: Healthcare professionals need quick answers to patient-specific questions
   - **Solution**: Conversational AI agent that answers questions based on patient context
   - **Benefit**: Instant access to patient information and clinical insights

### 4. **Template-Based Documentation**
   - **Problem**: Different insurance providers require different summary formats
   - **Solution**: Upload PDF templates to generate summaries matching specific requirements
   - **Benefit**: Ensures compliance with various documentation standards

### 5. **Knowledge Base Enhancement**
   - **Problem**: Generated summaries should improve future recommendations
   - **Solution**: Feedback loop that adds generated summaries back to the knowledge base
   - **Benefit**: Continuously improving system accuracy and relevance

## 🏗️ Architecture

```
┌─────────────┐         ┌──────────────┐         ┌─────────────┐
│  Streamlit  │ ──────> │   FastAPI    │ ──────> │   Groq API  │
│  Frontend   │  HTTP   │   Backend    │  HTTP   │   (LLM)     │
│  (Port 8501)│         │  (Port 8000) │         │  (Cloud)    │
└─────────────┘         └──────────────┘         └─────────────┘
                              │
                              ├──> MongoDB (Patient Records)
                              └──> ChromaDB (Vector Search)
```

## 🚀 Key Features

- **🤖 AI-Powered Summary Generation**: Uses Groq API with LLaMA 4 Maverick for fast, accurate summaries
- **💬 Conversational AI Agent**: Interactive chat interface for clinical queries
- **🔍 Semantic Case Search**: Find similar patient cases using Bio ClinicalBERT embeddings
- **📊 Modern UI**: Beautiful dark theme interface with smooth animations
- **⚡ High Performance**: FastAPI async backend for 40-60% faster responses
- **📄 Multiple Export Formats**: Download summaries as TXT, DOCX, or PDF
- **🎨 Template Support**: Custom PDF templates for different documentation requirements
- **💾 Feedback Loop**: Continuous learning from generated summaries

## 🛠️ Technology Stack

- **Frontend**: Streamlit with modern dark theme UI
- **Backend**: FastAPI with async/await for high performance
- **LLM**: Groq API with `meta-llama/llama-4-maverick-17b-128e-instruct`
- **Embeddings**: Bio ClinicalBERT (medical domain-specific)
- **Vector DB**: ChromaDB for similarity search
- **Database**: MongoDB for patient records
- **AI Agent**: AutoGen for conversational interface

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- Groq API key ([Get one here](https://console.groq.com/keys))
- MongoDB connection (cloud or local)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/rajiv-rane/clinical-summary-rag-v2.git
   cd clinical-summary-rag-v2
   ```

2. **Install dependencies**
   ```bash
   cd ingestion-phase
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the `ingestion-phase` directory:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

4. **Start the FastAPI backend**
   ```bash
   python start_api.py
   ```

5. **Start the Streamlit frontend** (in a new terminal)
   ```bash
   streamlit run app.py
   ```

6. **Access the application**
   Open your browser to `http://localhost:8501`

For detailed setup instructions, see [ingestion-phase/README.md](ingestion-phase/README.md)

## 📚 Documentation

- [Main README](ingestion-phase/README.md) - Complete setup and usage guide
- [FastAPI Documentation](ingestion-phase/README_FASTAPI.md) - Backend API details
- [Groq Setup Guide](ingestion-phase/GROQ_SETUP.md) - API key configuration

## 🔒 Security Considerations

- Never commit `.env` files containing API keys
- Use environment variables for sensitive configuration
- Implement authentication/authorization for production use
- Restrict CORS origins in production
- Enable HTTPS for production deployments

## 📝 License

[Add your license information here]

## 👥 Authors

- **Rajiv Rane** - [GitHub](https://github.com/rajiv-rane)

## 🙏 Acknowledgments

- Bio ClinicalBERT model by Emily Alsentzer
- Groq API for fast LLM inference
- FastAPI by Sebastián Ramírez
- Streamlit team
- ChromaDB team

---

**Version**: 2.0.0  
**Last Updated**: 2024

