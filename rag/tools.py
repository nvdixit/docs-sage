from langchain.tools import tool

from rag.vector_store import VectorStore

@tool("search_documents", return_direct=False)
def search_documents(query, k=1):
    """
    Tool to search for documents in the vector store that are similar to the given query.
    The agent will process the user's query into its most relevant form and use this tool to retrieve the most similar documents from the vector store.
    
    Args:
        query (str): The search query.
        k (int): The number of similar documents to retrieve.

    Returns:
        list: A list of Document objects that are similar to the query.
    """
    vs = VectorStore()
    matching_doc = vs.vector_store.similarity_search(query=query, k=k)
    return matching_doc

