# RAG-Based PDF Q&A Chatbot

## Project Overview

The RAG-Based PDF Q&A Chatbot is an AI-powered application that allows users to upload a PDF document and ask questions about its content.

The system uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from the uploaded PDF and provide accurate answers using a Large Language Model.

The chatbot also displays the relevant PDF page numbers used to generate the answer.

---

## Features

- Upload PDF documents
- Extract text from PDF pages
- Split extracted text into smaller chunks
- Generate text embeddings using Sentence Transformers
- Store embeddings in ChromaDB
- Retrieve relevant document chunks based on the user's question
- Generate answers using Google Gemini
- Display relevant PDF page numbers
- Interactive Streamlit web interface
- Secure API key management using `.env`

---

## Technologies Used

- Python 3.12
- Streamlit
- Sentence Transformers
- ChromaDB
- PyPDF
- Google Gemini API
- Python Dotenv

---

## RAG Architecture

The application follows the Retrieval-Augmented Generation pipeline:

```text
PDF Document
     |
     v
PDF Text Extraction
     |
     v
Text Chunking
     |
     v
Sentence Transformer
Embeddings
     |
     v
ChromaDB Vector Database
     |
     v
User Question
     |
     v
Question Embedding
     |
     v
Relevant Chunk Retrieval
     |
     v
Gemini LLM
     |
     v
Answer + Page References