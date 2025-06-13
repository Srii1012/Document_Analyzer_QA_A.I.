# Document-Research-Theme-Identification-Chatbot

Document Research Chatbot is a Generative AI application built with FastAPI and Streamlit for analyzing multiple documents, extracting answers with citations, and identifying common themes across documents. Users can upload 75+ documents (PDFs, images with OCR), ask questions, and receive comprehensive answers with document references.

## Table of Contents
- [About the Project](#about-the-project)
- [Code Structure & Demo Videos](#code-structure--demo-videos)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)

## About the Project
This AI-powered document research system revolutionizes how users interact with multiple documents. The application provides:
- Batch processing of 75+ documents (PDFs, images with OCR)
- Natural language question answering with precise citations
- Cross-document theme identification
- Synthesized summaries of key information
- Mock document generator for testing

## Code Structure & Demo Videos
document-research-chatbot/
          ├── app.py # Streamlit frontend
          ├── main.py # FastAPI backend
          ├── requirements.txt # Dependencies
          ├── data/ # Document storage
          └── qdrant_db/ # Vector database

Code Demo Video: https://drive.google.com/file/d/1nhu-hYdYEJaEhG8Mb1PhI7X49aRYznd9/view?usp=sharing
Appplication Demo Video: https://drive.google.com/file/d/1pT2kt7PgqrpBegZdkhYTZdjMaepCQt4v/view?usp=sharing

## Features
- **Multi-Document Processing**: Handle 75+ PDFs and images with OCR
- **Precise Citations**: Answers reference specific documents and pages
- **Theme Analysis**: AI identifies common themes across documents
- **Two-Part Results**:
  - Table of individual document answers
  - Synthesized summary with theme breakdown
- **Camera Integration**: Capture documents via mobile camera
- **Mock Data Generator**: Create sample documents for testing
- **Document Management**: View and manage uploaded files

## Tech Stack
**Frontend**: Streamlit  
**Backend**: FastAPI (Python)  
**Vector Database**: Qdrant  
**Embeddings**: all-MiniLM-L6-v2  
**LLM**: Llama3-70b (via Groq)  
**OCR**: Tesseract  
**PDF Processing**: pdfplumber  

## Installation

### Prerequisites
- Python 3.9+
- Groq API key
- Tesseract OCR installed

## Create and activate virtual environment:

python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

###Install dependencies:

pip install -r requirements.txt

###Set up environment variables:
Create .env file with:

GROQ_API_KEY=your_api_key_here

###Initialize Qdrant database:

mkdir -p qdrant_db
Run the application:

bash
# Start backend
uvicorn main:app --reload

# In another terminal, start frontend
streamlit run app.py

###Access the application at:

Frontend: http://localhost:8501

Backend API: http://localhost:8000

Usage
Upload Documents:

*Use the "Upload Files" tab to add documents
*Supported formats: PDF, PNG, JPG, JPEG
*Track progress in the sidebar

Ask Questions:

*Enter your query in natural language
*Select search mode (All Documents/Smart Themes/Detailed Citations)

Review Results:

*View individual document answers in table format
*Examine synthesized themes and summaries
*Check confidence scores and metrics

Advanced Features:

*Use camera capture for mobile document processing
*Generate mock documents for testing
*Manage documents in File Organizer tab
