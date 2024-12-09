# rag_service.py
import pyarrow as pa
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import logging
from .db_config import collection, model
from .langchain_pipeline import LangChainRetrievalPipeline
from .embed_documents import process_documents
from langchain.llms import OpenAI



# Initialize FastAPI app, model, and LanceDB
app = FastAPI()

# Initialize logging
logging.basicConfig(level=logging.INFO)  # Set logging level to INFO

# Initialize the LangChain pipeline
# langchain_pipeline = LangChainRetrievalPipeline(collection)

# Define request and response models
class TextRequest(BaseModel):
    text: str

class RetrievalRequest(BaseModel):
    query: str
    api_key: str  # Add API key to the request model

class EnhancedPromptResponse(BaseModel):
    enhanced_prompt: str
    llm_response: str
    documents_used: List[Dict[str, Any]]

@app.post("/embed/")
def create_embedding(request: TextRequest):
    embedding = model.encode(request.text).tolist()  # Convert to list for JSON compatibility
    return {"embedding": embedding}

@app.post("/store/")
def store_embedding(request: TextRequest):
    try:
        embedding = model.encode(request.text).tolist()  # Convert to list for storage
        data = {
            "text_id": request.text,
            "vector": embedding,
            "original_text": request.text
        }
        logging.info(f"Storing data in the collection: {data}")
        collection.add([data])  # Use add method with a list of dictionaries
        return {"status": "stored", "text_id": request.text}
    except Exception as e:
        logging.error(f"Error in store_embedding: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# **TODO 1**: Implement the `retrieve_and_enhance` function to process an incoming query, retrieve contextually relevant documents, enhance the prompt, and fetch a response from the language model.
# ```plaintext
# pseudocode:
# 1. **Initialize Language Model**:
#    - Extract the API key from the incoming request data to authenticate with the language model.
#    - Create an `llm_model` instance using the extracted API key, enabling the function to query the model securely.
# 2. **Set Up Retrieval Pipeline**:
#    - Instantiate a `LangChainRetrievalPipeline` with two main components:
#      - `collection`, which references the document storage system for retrieving contextually relevant documents.
#      - `llm_model`, the language model that will generate responses based on enhanced prompts.
# 3. **Retrieve and Enhance Prompt**:
#    - Call the `retrieve_and_enhance` method of `langchain_pipeline`, passing in the query from the request data.
#    - This method retrieves the most relevant documents, formats them into a coherent context, and combines them with the user query to create an enhanced prompt.
# 4. **Return Structured Response**:
#    - Package the output from `retrieve_and_enhance` into a structured response containing:
#      - `enhanced_prompt`: the modified prompt incorporating relevant document context.
#      - `llm_response`: the generated response from the language model.
#      - `documents_used`: metadata on the documents used to create context, aiding in traceability and context clarity.
# 5. **Error Handling**:
#    - Enclose the entire logic in a try-except block to handle unexpected issues.
#    - Log errors and raise an HTTP 500 error with details to ensure troubleshooting and debugging are efficient.
# ```

def retrieve_and_enhance(query: str, api_key: str) -> EnhancedPromptResponse:
    try:
        # Initialize the Language Model
        llm_model = OpenAI(api_key)

        # Set Up Retrieval Pipeline
        langchain_pipeline = LangChainRetrievalPipeline(collection, llm_model)

        # Retrieve and Enhance Prompt
        enhanced_prompt, llm_response, documents_used = langchain_pipeline.retrieve_and_enhance(query)

        # Return Structured Response
        return EnhancedPromptResponse(enhanced_prompt=enhanced_prompt, llm_response=llm_response, documents_used=documents_used)
    except Exception as e:
        logging.error(f"Error in retrieve_and_enhance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embed_all_documents/")
def embed_all_documents():
    process_documents()  # Call the function from embed_documents.py
    return {"status": "Documents embedded and stored successfully"}

@app.post("/clear_embeddings/")
def clear_embeddings():
    try:
        collection.delete(where="True")  # "True" matches all rows, effectively clearing the table
        return {"status": "Embeddings cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Debugging endpoints

@app.get("/check_embeddings/")
def check_embeddings():
    try:
        # Use count_rows to count the records in the collection
        num_embeddings = collection.count_rows()
        return {"number_of_embeddings": num_embeddings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sample_embedding/")
def sample_embedding():
    try:
        # Use scanner to retrieve a sample of records
        sample = []
        for i, record in enumerate(collection.scanner()):
            if i >= 5:  # Limit to first 5 records for sampling
                break
            sample.append(record)
        return {"sample_embeddings": sample}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 

