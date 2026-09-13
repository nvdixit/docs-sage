from rag.vector_store import VectorStore

vs = VectorStore()

results = vs.vector_store.similarity_search(query="github pages", k=1)
for doc in results:
    print(f"* {doc.page_content} [{doc.metadata}]")