from pymongo import MongoClient
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


class VectorStore:

    MONGO_CONNECTION_STRING = "mongodb://localhost:27017/"

    def __init__(self, mongo_connection_string=MONGO_CONNECTION_STRING):
        '''
        Initializes the VectorStore class by connecting to the MongoDB database and creating a vector store from the processed documents.'''
        self.mongo_connection_string = mongo_connection_string
        self.client = MongoClient(self.mongo_connection_string)
        self.db = self.client["docs-sage"]
        self.collection = self.db["processed-data"]
        self.vector_store = None
        self.create_vector_store()


    def create_vector_store(self):
        '''
        Creates a vector store from the processed documents in the MongoDB collection.
        It retrieves all documents from the collection, creates Document objects for each document's text content,
        and then uses the OllamaEmbeddings model to generate embeddings for the documents.
        Finally, it creates a Chroma vector store from the documents and their embeddings.'''

        # Pull all documents from the processed-data collection in the docs-sage database
        all_docs = list(self.collection.find({}))

        chroma_docs = []
        ids = []
        idx = 0
        for doc in all_docs:
            # Create a Document object for each document's text content
            doc = Document(page_content=doc["content"], metadata={"summary": doc["summary"]})
            chroma_docs.append(doc)
            ids.append(str(idx))
            idx += 1

        # Generate embeddings for the documents using the OllamaEmbeddings model nomic-embed-text
        # and create a Chroma vector store
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vector_store = Chroma.from_documents(
            documents=chroma_docs,
            collection_name="embedded_docs",
            embedding=embeddings,
            ids=ids,
        )

        # Set the vector_store attribute to the created Chroma vector store
        self.vector_store = vector_store