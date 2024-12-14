# embed_documents.py
import os
import json
import requests
from sentence_transformers import SentenceTransformer
import logging
from db_config import collection, model


# Ensure logging is set up
logging.basicConfig(level=logging.INFO)

# Directory containing the HR documents
base_dir = "../generated_hr_docs"

# API endpoint for storing embeddings
store_endpoint = "http://127.0.0.1:8000/store/"

def embed_and_store_document(text, text_id):
    # Generate the embedding vector
    embedding_vector = model.encode(text).tolist()
    
    # Format the data for storing
    store_data = {
        "text_id": text_id,
        "vector": embedding_vector,
        "original_text": text
    }
    
    # Send the data to the /store/ endpoint
    response = requests.post(store_endpoint, json=store_data)
    
    if response.status_code == 200:
        print(f"Document {text_id} stored successfully.")
    else:
        print(f"Error storing document {text_id}: {response.text}")


def process_documents():

    try:
        for subdir, _, files in os.walk(base_dir):
            for file in files:
                if file.endswith(".json"):
                    file_path = os.path.join(subdir, file)
                    
                    # Read the JSON document
                    with open(file_path, "r", encoding="utf-8") as f:
                        document = json.load(f)
                        text_content = document.get("content", "")
                        text_id = os.path.splitext(file)[0]
                        
                        if not text_content:
                            logging.warning(f"No 'content' found in {file_path}")
                            continue
                        
                        # Generate embedding
                        embedding = model.encode(text_content).tolist()
                        logging.info(f"Embedding generated for {file_path} with length {len(embedding)}")

                        # Store embedding in LanceDB with error handling
                        data = {
                            "text_id": text_id,
                            "vector": embedding,
                            "original_text": text_content[:200]  # Store a snippet of the content
                        }
                        
                        try:
                            collection.add([data])
                            logging.info(f"Document {file_path} stored successfully in LanceDB.")
                            
                            # Check the number of rows in LanceDB after adding
                            current_count = collection.count_rows()
                            logging.info(f"Current number of embeddings in collection after adding: {current_count}")

                        except Exception as e:
                            logging.error(f"Error adding data to collection: {str(e)}")
                            continue

        logging.info("All documents processed successfully.")

    except Exception as e:
        logging.error(f"Error processing documents: {str(e)}")



# Run the process
if __name__ == "__main__":
    process_documents()
