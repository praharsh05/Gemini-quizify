from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents import Document
import streamlit as st


class ChromaCollectionCreator:
    def __init__(self, processor, embed_model):
        """
        Initializes the ChromaCollectionCreator with a DocumentProcessor instance and embeddings
          configuration.
        :param processor: An instance of DocumentProcessor that has processed documents.
        :param embeddings_config: An embedding client for embedding documents.
        """
        self.processor = processor      # This will hold the DocumentProcessor
        self.embed_model = embed_model  # This will hold the EmbeddingClient
        self.db = None                  # This will hold the Chroma collection

    def create_chroma_collection(self):
        """
        Task: Create a Chroma collection from the documents processed by the DocumentProcessor 
        instance.

        Steps:
        1. Check if any documents have been processed by the DocumentProcessor instance.
          If not, display an error message using streamlit's error widget.

        2. Split the processed documents into text chunks suitable for embedding and indexing.
          Use the CharacterTextSplitter from Langchain to achieve this. You'll need to define a 
          separator, chunk size, and chunk overlap.
          https://python.langchain.com/docs/modules/data_connection/document_transformers/

        3. Create a Chroma collection in memory with the text chunks obtained from step 2 and 
          the embeddings model initialized in the class. Use the Chroma.from_documents method for 
          this purpose.
          https://python.langchain.com/docs/integrations/vectorstores/chroma#use-openai-embeddings
          https://docs.trychroma.com/getting-started

        Instructions:
        - Begin by verifying that there are processed pages available. If not, inform the user that
          no documents are found.

        - If documents are available, proceed to split these documents into smaller text chunks. This
          operation prepares the documents for embedding and indexing. Look into using the 
          CharacterTextSplitter with appropriate parameters (e.g., separator, chunk_size, chunk_overlap).

        - Next, with the prepared texts, create a new Chroma collection. This step involves using the 
          embeddings model (self.embed_model) along with the texts to initialize the collection.

        - Finally, provide feedback to the user regarding the success or failure of the Chroma 
          collection creation.
        """

        # Step 1: Check for processed documents
        if len(self.processor.pages) == 0:
            st.error("No documents found!", icon="🚨")
            return

        # Step 2: Split documents into text chunks
        # Use a TextSplitter from Langchain to split the documents into smaller text chunks
        # https://python.langchain.com/docs/modules/data_connection/document_transformers/character_text_splitter

        # Assuming 'processed_documents' is a list of documents
        text_splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=1000,
            chunk_overlap=200
        )

        text_chunks = text_splitter.split_documents(self.processor.pages)
        if self.processor.pages is not None:
            st.success(
                f"Successfully split pages to {len(text_chunks)} documents!", icon="✅")

        # Output the text chunks
        for i, chunk in enumerate(text_chunks):
            print(f"Chunk {i + 1}: {chunk}")

        # Step 3: Create the Chroma Collection
        # https://docs.trychroma.com/
        # Create a Chroma in-memory client using the text chunks and the embeddings model
        # Assuming 'embed_model' is the initialized embeddings model
        self.db = Chroma.from_documents(text_chunks, self.embed_model)

        # Optionally, print a confirmation message
        print("Chroma collection created successfully!")

        if self.db:  # Assuming chroma_collection is the created collection
            st.success("Chroma Collection created successfully!", icon="✅")
        else:
            st.error("Failed to create Chroma Collection.", icon="🚨")

    def query_chroma_collection(self, query) -> Document:
        """
        Queries the created Chroma collection for documents similar to the query.
        :param query: The query string to search for in the Chroma collection.

        Returns the first matching document from the collection with similarity score.
        """
        if self.db:
            docs = self.db.similarity_search_with_relevance_scores(query)
            if docs:
                return docs[0]
            else:
                st.error("No matching documents found!", icon="🚨")
        else:
            st.error("Chroma Collection has not been created!", icon="🚨")
