import os
import json
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from fastapi import FastAPI, Request, Form, Response
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http.models import PointStruct, VectorParams
from fastapi.middleware.cors import CORSMiddleware

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up templates and static files
templates = Jinja2Templates(directory="template")
app.mount("/static", StaticFiles(directory="static"), name="static")

# LLM initialization
local_llm = os.getenv("LLM_PATH", "C:/Users/prish/Downloads/BioMistral-7B.Q4_K_M.gguf")
llm = LlamaCpp(
    model_path=local_llm,
    temperature=0.3,
    max_tokens=5000,
    top_p=1
)
logging.info("LLM Initialized....")

# Prompt template definition
prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Context: {context}
Question: {question}
Only return the helpful answer. Answer must be detailed and well explained.
Helpful answer:
"""

# Qdrant client setup
client = QdrantClient(url="http://localhost:6333")
collection_name = "vector_db"

# Create collection if it doesn't exist
try:
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=768, distance='Cosine')
    )
    logging.info(f"Collection '{collection_name}' created successfully.")
except Exception as e:
    logging.error(f"Error creating collection: {e}")

# Embeddings initialization using Sentence Transformers
embedding_model = SentenceTransformer('dmis-lab/biobert-base-cased-v1.1')

def set_embeddings(sentences):
    try:
        embeddings = embedding_model.encode(sentences, show_progress_bar=True).astype(np.float32)  # Ensure float32
        logging.info(f"Embeddings type: {type(embeddings)}, Embeddings shape: {embeddings.shape}")  # Debugging info
        points = [
            PointStruct(id=i, vector=embedding.tolist(), payload={"sentence": sentence})
            for i, (embedding, sentence) in enumerate(zip(embeddings, sentences))
        ]
        client.upsert(collection_name=collection_name, points=points)
        logging.info(f'Embeddings for {len(sentences)} sentences stored in Qdrant collection "{collection_name}".')
    except Exception as e:
        logging.error(f"Error setting embeddings: {e}")

if __name__ == "__main__":
    sentences = [
        "This is a sample sentence.",
        "Here is another sentence for embedding.",
        "Qdrant is a vector search engine."
    ]
    set_embeddings(sentences)

# Setup retriever
db = QdrantVectorStore(client=client, collection_name=collection_name, embedding=embedding_model)
retriever = db.as_retriever(search_kwargs={"k": 3})
prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

# Define the RetrievalQA chain
qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever, 
    return_source_documents=True, 
    chain_type_kwargs={"prompt": prompt}, 
    verbose=True
)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("frontend.html", {"request": request})

@app.post("/get_response")
async def get_response(query: str = Form(...)):
    try:
        logging.info(f"Received query: {query}")
        response = qa(query)
        answer = response['result']
        source_document = response['source_documents'][0].page_content
        doc = response['source_documents'][0].metadata['source']
        response_data = jsonable_encoder({"answer": answer, "source_document": source_document, "doc": doc})
        return Response(content=json.dumps(response_data), media_type="application/json")
    except Exception as e:
        logging.error(f"Error processing query: {e}")
        return Response(content=json.dumps({"error": "Error processing your request"}), media_type="application/json", status_code=500)

def query_embeddings(query, top_k=3):
    try:
        query_embedding = embedding_model.encode(query).astype(np.float32).tolist()  # Ensure float32 conversion
        logging.info(f"Query embedding type: {type(query_embedding)}")  # Debugging information
        search_results = client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=top_k
        )
        for result in search_results:
            logging.info(f"ID: {result.id}, Score: {result.score}, Sentence: {result.payload['sentence']}")
    except Exception as e:
        logging.error(f"Error querying embeddings: {e}")

# Example usage for querying
if __name__ == "__main__":
    query_embeddings("What is Parkinson's?", top_k=3)





