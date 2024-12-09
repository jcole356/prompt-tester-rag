# langchain_pipeline.py
from sentence_transformers import SentenceTransformer
from langchain.prompts import PromptTemplate

# **TODO 1**: Set up the `SentenceTransformer` model to generate embeddings and build a prompt template for enhancing user prompts with contextual information.
# ```plaintext
# pseudocode:
# 1. Initialize the sentence embedding model by loading the "all-MiniLM-L6-v2" configuration, specifically optimized for generating compact embeddings.
# 2. Define a prompt template that combines a query with additional contextual information:
#     - Specify placeholders for "query" and "context" to dynamically fill with input text and related context.
#     - Format the template to place the query at the top, followed by "Contextual Info:" and the context content on a new line.
# ```

# Initialize the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

template = """
   {query}
   Contextual Info: {context}
"""

# Define a prompt template for enhancing user queries
prompt_template = PromptTemplate(
   input_variables=["query", "context"],
   template=template
)

class LangChainRetrievalPipeline:
    # **TODO 2**: Define the LangChain Retrieval Pipeline class to manage document retrieval and prompt enhancement for generating relevant responses from a language model.
    # ```plaintext
    # pseudocode:
    # 1. Define a class named `LangChainRetrievalPipeline` to retrieve documents and enhance prompts. This class will initialize with two parameters:
    #    - `collection`: A collection (or database) where documents are stored and can be queried.
    #    - `llm_model`: A language model (LLM) instance that will generate responses based on the enhanced prompt.
    # 2. Inside the `__init__` method:
    #    - Assign `collection` and `llm_model` as instance variables for later access in other class methods.
    # 3. Define a method named `retrieve_and_enhance` that takes `query` as an input. This method will:
    #    - **Generate an embedding for the query** using the `model.encode(query)` method and convert it to a list. This embedding represents the meaning of the query.
    #    - **Retrieve matching documents** by searching the `collection` with the query embedding, limiting the results to the top 5 most relevant documents.
    #    - **Extract relevant text snippets** (context) from each retrieved document, limiting each snippet to the first 200 characters. Concatenate these snippets into a single string `context_texts`, which will provide background for the query.
    #    - **Format an enhanced prompt** by combining the `query` and `context_texts` within a pre-defined `prompt_template`. This enhanced prompt offers richer context for the LLM to generate more precise responses.
    # 4. Use the `llm` instance to pass the `enhanced_prompt` to the LLM and capture the response in `llm_response`.
    # 5. **Prepare document metadata** for each retrieved document:
    #    - Create a dictionary with `text_id`, a snippet of `original_text`, and `score` for each result. This metadata provides reference and context for the LLM’s response.
    # 6. Return a dictionary containing:
    #    - `enhanced_prompt`: The prompt with added context used in the query.
    #    - `llm_response`: The language model’s response to the enhanced prompt.
    #    - `documents_used`: Metadata of the documents retrieved, providing context for the LLM response.
    # ```
    
    def __init__(self, collection, llm_model):
        self.collection = collection
        self.llm_model = llm_model

    def retrieve_and_enhance(self, query):
        # Generate an embedding for the query
        query_embedding = model.encode(query).tolist()

        # Retrieve matching documents
        retrieved_docs = self.collection.search(query_embedding, top_k=5)

        # Extract relevant text snippets from each document
        context_texts = ""
        for doc in retrieved_docs:
            context_texts += doc["original_text"][:200] + "..."  # Limit to 200 characters

        # Format an enhanced prompt
        enhanced_prompt = prompt_template.format(query=query, context=context_texts)

        # Generate a response from the language model
        llm_response = self.llm_model.generate_response(enhanced_prompt)

        # Prepare document metadata
        documents_used = []
        for doc in retrieved_docs:
            doc_metadata = {
                "text_id": doc["text_id"],
                "original_text": doc["original_text"],
                "score": doc["score"]
            }
            documents_used.append(doc_metadata) # Add metadata to the list

        return {
            "enhanced_prompt": enhanced_prompt,
            "llm_response": llm_response,
            "documents_used": documents_used
        }
    