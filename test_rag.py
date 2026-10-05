from rag_pipeline import (
    extract_text_from_pdf,
    create_chunks,
    create_embeddings,
    store_in_chromadb,
    retrieve_documents,
    generate_answer
)


PDF_PATH = "data/sample.pdf"


print("=" * 60)
print("RAG PDF Q&A CHATBOT TEST")
print("=" * 60)


# ============================================================
# STEP 1: READ PDF
# ============================================================

print("\n1. Reading PDF...")

pages = extract_text_from_pdf(PDF_PATH)

print(
    f"Total pages: {len(pages)}"
)


# ============================================================
# STEP 2: CREATE CHUNKS
# ============================================================

print("\n2. Creating text chunks...")

chunks = create_chunks(pages)

print(
    f"Total chunks: {len(chunks)}"
)


# ============================================================
# STEP 3: CREATE EMBEDDINGS
# ============================================================

print("\n3. Creating embeddings...")

embeddings = create_embeddings(chunks)

print(
    f"Embedding shape: {embeddings.shape}"
)


# ============================================================
# STEP 4: STORE IN CHROMADB
# ============================================================

print("\n4. Storing data in ChromaDB...")

collection = store_in_chromadb(
    chunks,
    embeddings
)

print(
    f"Stored documents: {collection.count()}"
)


# ============================================================
# STEP 5: TEST RETRIEVAL
# ============================================================

question = "What does Career GPS AI do?"

print(
    f"\n5. Testing retrieval with question:"
)

print(
    f"   {question}"
)


results = retrieve_documents(
    question,
    collection,
    top_k=3
)


print("\nRetrieved information:")


for i, document in enumerate(
    results["documents"][0]
):

    page = results["metadatas"][0][i]["page"]

    print(
        f"\n--- Result {i + 1} "
        f"(Page {page}) ---"
    )

    print(
        document[:500]
    )


# ============================================================
# STEP 6: GENERATE ANSWER
# ============================================================

print(
    "\n6. Generating answer using Gemini..."
)


answer = generate_answer(
    question,
    results
)


# ============================================================
# STEP 7: DISPLAY ANSWER
# ============================================================

print("\n" + "=" * 60)

print("ANSWER")

print("=" * 60)

print(answer)

print("\n" + "=" * 60)

print("RAG TEST COMPLETED SUCCESSFULLY")

print("=" * 60)