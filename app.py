import os
import time

import streamlit as st
import chromadb
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from google import genai


# --------------------------------------------------
# Load environment variables
# --------------------------------------------------

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("GEMINI_API_KEY is missing. Please check your .env file.")
    st.stop()


# --------------------------------------------------
# Gemini client
# --------------------------------------------------

gemini_client = genai.Client(api_key=api_key)


# --------------------------------------------------
# Streamlit page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="RAG-Based PDF Q&A Chatbot",
    page_icon="📄",
    layout="wide"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("📄 RAG-Based PDF Q&A Chatbot")

st.write(
    "Upload a PDF and ask questions about its contents."
)


# --------------------------------------------------
# Constants
# --------------------------------------------------

DATA_FOLDER = "data"
PDF_PATH = os.path.join(
    DATA_FOLDER,
    "uploaded_document.pdf"
)

CHROMA_PATH = "./chroma_db"

COLLECTION_NAME = "pdf_documents"


# --------------------------------------------------
# Create data folder
# --------------------------------------------------

os.makedirs(DATA_FOLDER, exist_ok=True)


# --------------------------------------------------
# Load embedding model
# --------------------------------------------------

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()


# --------------------------------------------------
# Connect to ChromaDB
# --------------------------------------------------

@st.cache_resource
def get_chroma_client():

    return chromadb.PersistentClient(
        path=CHROMA_PATH
    )


chroma_client = get_chroma_client()


# --------------------------------------------------
# Extract PDF text
# --------------------------------------------------

def extract_pdf_text(pdf_path):

    reader = PdfReader(pdf_path)

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text()

        if text:

            pages.append(
                {
                    "page": page_number,
                    "text": text
                }
            )

    return pages


# --------------------------------------------------
# Create text chunks
# --------------------------------------------------

def create_chunks(
    pages,
    chunk_size=500,
    overlap=50
):

    chunks = []

    for page_data in pages:

        text = page_data["text"]

        page_number = page_data["page"]

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[start:end]

            if chunk_text.strip():

                chunks.append(
                    {
                        "text": chunk_text,
                        "page": page_number
                    }
                )

            start += chunk_size - overlap

    return chunks


# --------------------------------------------------
# Store documents in ChromaDB
# --------------------------------------------------

def store_in_chroma(chunks):

    # Delete only the collection.
    # Do NOT delete the entire chroma_db folder.

    try:

        chroma_client.delete_collection(
            name=COLLECTION_NAME
        )

    except Exception:

        pass

    collection = chroma_client.create_collection(
        name=COLLECTION_NAME
    )

    documents = [
        chunk["text"]
        for chunk in chunks
    ]

    metadatas = [
        {
            "page": chunk["page"]
        }
        for chunk in chunks
    ]

    ids = [
        f"chunk_{i}"
        for i in range(len(chunks))
    ]

    embeddings = embedding_model.encode(
        documents
    ).tolist()

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings
    )

    return collection


# --------------------------------------------------
# Retrieve relevant chunks
# --------------------------------------------------

def retrieve_documents(
    collection,
    question,
    top_k=3
):

    question_embedding = embedding_model.encode(
        [question]
    ).tolist()

    results = collection.query(
        query_embeddings=question_embedding,
        n_results=top_k
    )

    return results


# --------------------------------------------------
# Generate answer using Gemini
# --------------------------------------------------

def generate_answer(
    question,
    results
):

    documents = results["documents"][0]

    metadatas = results["metadatas"][0]

    context_parts = []

    for document, metadata in zip(
        documents,
        metadatas
    ):

        context_parts.append(
            f"Page {metadata['page']}:\n{document}"
        )

    context = "\n\n".join(
        context_parts
    )

    prompt = f"""
You are a helpful PDF question-answering assistant.

Answer the user's question using ONLY the information
provided in the PDF context below.

Do not use outside knowledge.

If the answer cannot be found in the PDF context, say:

"I couldn't find sufficient information about this in the uploaded document."

Always mention the relevant PDF page number or page numbers.

PDF CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""

    # --------------------------------------------------
    # Try Gemini 3.8 Flash
    # --------------------------------------------------

    models_to_try = [
        "gemini-3.8-flash",
        "gemini-3.5-flash-lite"
    ]

    last_error = None

    for model_name in models_to_try:

        for attempt in range(2):

            try:

                response = gemini_client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )

                return response.text

            except Exception as error:

                last_error = error

                error_text = str(error)

                # Retry temporary 503 / unavailable errors

                if (
                    "503" in error_text
                    or "UNAVAILABLE" in error_text
                    or "high demand" in error_text
                ):

                    time.sleep(
                        2 * (attempt + 1)
                    )

                else:

                    break

    raise Exception(
        "Gemini is temporarily unavailable. "
        "Please try your question again in a few seconds.\n\n"
        f"Last error: {last_error}"
    )


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.header("📂 Upload PDF")

uploaded_file = st.sidebar.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)


# --------------------------------------------------
# Process PDF
# --------------------------------------------------

if uploaded_file is not None:

    if st.sidebar.button(
        "Process PDF"
    ):

        with st.spinner(
            "Processing PDF..."
        ):

            # Save uploaded PDF

            with open(
                PDF_PATH,
                "wb"
            ) as file:

                file.write(
                    uploaded_file.getbuffer()
                )

            # Extract text

            pages = extract_pdf_text(
                PDF_PATH
            )

            if not pages:

                st.error(
                    "Could not extract text from the PDF."
                )

                st.stop()

            # Create chunks

            chunks = create_chunks(
                pages
            )

            # Store embeddings

            collection = store_in_chroma(
                chunks
            )

            # Save collection in session

            st.session_state.collection_ready = True

            st.session_state.pdf_name = (
                uploaded_file.name
            )

            st.success(
                "PDF processed successfully!"
            )

            st.info(
                f"Pages: {len(pages)} | "
                f"Chunks: {len(chunks)}"
            )


# --------------------------------------------------
# Check whether collection exists
# --------------------------------------------------

collection_ready = st.session_state.get(
    "collection_ready",
    False
)


# --------------------------------------------------
# Status message
# --------------------------------------------------

if collection_ready:

    st.success(
        "✅ PDF is ready. You can ask questions below."
    )

else:

    st.info(
        "👈 Upload a PDF and click "
        "**Process PDF** to begin."
    )


# --------------------------------------------------
# Chat history
# --------------------------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = []


# --------------------------------------------------
# Display previous messages
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# --------------------------------------------------
# Chat input
# --------------------------------------------------

question = st.chat_input(
    "Ask a question about your PDF..."
)


# --------------------------------------------------
# Process user question
# --------------------------------------------------

if question:

    if not collection_ready:

        st.warning(
            "Please upload and process a PDF first."
        )

        st.stop()

    # Display user question

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    # Get Chroma collection

    collection = chroma_client.get_collection(
        name=COLLECTION_NAME
    )

    # Generate answer

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching the PDF and generating answer..."
        ):

            try:

                results = retrieve_documents(
                    collection,
                    question,
                    top_k=3
                )

                answer = generate_answer(
                    question,
                    results
                )

                st.markdown(answer)

                # Show source pages

                source_pages = sorted(
                    set(
                        metadata["page"]
                        for metadata
                        in results["metadatas"][0]
                    )
                )

                st.caption(
                    "📚 Source pages: "
                    + ", ".join(
                        f"Page {page}"
                        for page in source_pages
                    )
                )

                # Save assistant response

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

            except Exception as error:

                st.error(
                    str(error)
                )