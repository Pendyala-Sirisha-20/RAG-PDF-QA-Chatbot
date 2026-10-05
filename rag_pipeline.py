from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from dotenv import load_dotenv
from google import genai
import os


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY not found in .env file. "
        "Please add your Gemini API key."
    )

# Gemini client
client = genai.Client(api_key=api_key)


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_text_from_pdf(pdf_path):
    """
    Extract text from every page of the PDF.
    """

    reader = PdfReader(pdf_path)

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text() or ""

        pages.append({
            "page": page_number,
            "text": text
        })

    return pages


# ============================================================
# TEXT CHUNKING
# ============================================================

def create_chunks(
    pages,
    chunk_size=500,
    overlap=50
):
    """
    Split PDF text into smaller chunks while
    preserving the page number.
    """

    chunks = []

    for page in pages:

        text = page["text"]

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[
                start:end
            ].strip()

            if chunk_text:

                chunks.append({
                    "text": chunk_text,
                    "page": page["page"]
                })

            start += chunk_size - overlap

    return chunks


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

def create_embeddings(chunks):
    """
    Convert text chunks into vector embeddings
    using Sentence Transformers.
    """

    print("Loading embedding model...")

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(texts)

    return embeddings


# ============================================================
# STORE DATA IN CHROMADB
# ============================================================

def store_in_chromadb(
    chunks,
    embeddings
):
    """
    Store document chunks, embeddings and page numbers
    in ChromaDB.
    """

    chroma_client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    collection = chroma_client.get_or_create_collection(
        name="pdf_documents"
    )

    ids = [
        f"chunk_{i}"
        for i in range(len(chunks))
    ]

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

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )

    return collection


# ============================================================
# RETRIEVE RELEVANT DOCUMENTS
# ============================================================

def retrieve_documents(
    question,
    collection,
    top_k=3
):
    """
    Find the most relevant PDF chunks for the question.
    """

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    question_embedding = model.encode(
        [question]
    )

    results = collection.query(
        query_embeddings=question_embedding.tolist(),
        n_results=top_k
    )

    return results


# ============================================================
# GENERATE ANSWER USING GEMINI
# ============================================================

def generate_answer(
    question,
    results
):
    """
    Generate an answer using only the retrieved
    information from the PDF.
    """

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

Your job is to answer the user's question using ONLY
the information provided in the PDF context.

Do not use outside knowledge.

If the answer cannot be found in the provided PDF context,
say exactly:

"I couldn't find sufficient information about this in the uploaded document."

Always mention the relevant PDF page number or page numbers.

PDF CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""

    # ========================================================
    # TRY GEMINI 3.8 FLASH
    # ========================================================

    try:

        print(
            "\nTrying Gemini 3.8 Flash..."
        )

        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt
        )

        return response.text

    except Exception as first_error:

        print(
            "\nGemini 3.8 Flash is temporarily unavailable."
        )

        print(
            "Trying Gemini 3.5 Flash-Lite..."
        )

        # ====================================================
        # TRY FALLBACK MODEL
        # ====================================================

        try:

            response = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=prompt
            )

            return response.text

        except Exception as second_error:

            raise Exception(
                "Both Gemini models are temporarily "
                "unavailable. Please try again later.\n\n"
                f"Main model error: {first_error}\n"
                f"Fallback model error: {second_error}"
            )


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    pdf_path = "data/sample.pdf"

    print("=" * 60)

    print(
        "RAG-BASED PDF Q&A CHATBOT"
    )

    print("=" * 60)


    # --------------------------------------------------------
    # STEP 1: READ PDF
    # --------------------------------------------------------

    print(
        "\n1. Reading PDF..."
    )

    pages = extract_text_from_pdf(
        pdf_path
    )

    print(
        f"Total pages: {len(pages)}"
    )


    # --------------------------------------------------------
    # STEP 2: CREATE CHUNKS
    # --------------------------------------------------------

    print(
        "\n2. Creating text chunks..."
    )

    chunks = create_chunks(
        pages
    )

    print(
        f"Total chunks: {len(chunks)}"
    )


    # --------------------------------------------------------
    # STEP 3: CREATE EMBEDDINGS
    # --------------------------------------------------------

    print(
        "\n3. Creating embeddings..."
    )

    embeddings = create_embeddings(
        chunks
    )

    print(
        f"Embedding shape: {embeddings.shape}"
    )


    # --------------------------------------------------------
    # STEP 4: STORE IN CHROMADB
    # --------------------------------------------------------

    print(
        "\n4. Storing data in ChromaDB..."
    )

    collection = store_in_chromadb(
        chunks,
        embeddings
    )

    print(
        f"Stored documents: {collection.count()}"
    )

    print(
        "\nRAG database created successfully!"
    )


    # --------------------------------------------------------
    # STEP 5: ASK QUESTION
    # --------------------------------------------------------

    question = input(
        "\nAsk a question about the PDF: "
    )


    # --------------------------------------------------------
    # STEP 6: RETRIEVE RELEVANT CHUNKS
    # --------------------------------------------------------

    print(
        "\n5. Searching the PDF..."
    )

    results = retrieve_documents(
        question,
        collection
    )


    # --------------------------------------------------------
    # STEP 7: GENERATE ANSWER
    # --------------------------------------------------------

    print(
        "\n6. Generating answer using Gemini..."
    )

    answer = generate_answer(
        question,
        results
    )


    # --------------------------------------------------------
    # STEP 8: DISPLAY ANSWER
    # --------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "ANSWER"
    )

    print(
        "=" * 60
    )

    print(answer)

    print(
        "\n" + "=" * 60
    )