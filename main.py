import os
import logging
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import UnstructuredFileLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        embeddings = HuggingFaceEmbeddings(model_name='dmis-lab/biobert-base-cased-v1.1')
        logger.info("Embeddings initialized.")

        loader = DirectoryLoader('data/', glob="**/*.pdf", show_progress=True, loader_cls=UnstructuredFileLoader)
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} documents.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=70)
        texts = text_splitter.split_documents(documents)
        logger.info(f"Split documents into {len(texts)} chunks.")

        url = "http://localhost:6333"
        qdrant = Qdrant.from_documents(
            texts,
            embeddings,
            url=url,
            prefer_grpc=False,
            collection_name="vector_db"
        )

        logger.info("Vector DB successfully created!")
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
