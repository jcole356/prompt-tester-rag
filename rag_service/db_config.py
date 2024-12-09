# db_config.py
import lancedb
import logging
from sentence_transformers import SentenceTransformer
import pyarrow as pa

# Initialize logging
logging.basicConfig(level=logging.INFO)

# **TODO 1**: Initialize the `SentenceTransformer` model, connect to LanceDB, define the embedding dimensions, and set up the schema for storing embeddings.
# ```plaintext
# pseudocode:
# 1. Initialize the sentence embedding model by loading a specific model, 'all-MiniLM-L6-v2', using `SentenceTransformer`.
# 2. Connect to LanceDB by defining the directory location where the database is stored.
# 3. Set the embedding dimensions, specifying the fixed size for each vector (e.g., 384 dimensions).
# 4. Define the schema for the embeddings table:
#    - Add a unique identifier `text_id` as a string.
#    - Set the `vector` field as a list of floats with the specified embedding dimension.
#    - Add `original_text` as a string field to store the raw document content.
# ```

# Initialize the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Connect to LanceDB
db = lancedb.connect("lance_db")

# Define the embedding dimensions
embedding_dim = 384

# Define the schema for the embeddings table
schema = pa.schema([
    pa.field("text_id", pa.string()),
    pa.field("vector", pa.list_(pa.float32()), nullable=False),
    pa.field("original_text", pa.string())
])

# **TODO 2**: Initialize LanceDB by creating an embeddings table if it doesn’t exist, or opening the existing table. Configure the table to store `text_id`, `vector`, and `original_text` fields using the specified schema.
# ```plaintext
# pseudocode:
# 1. Check if an "embeddings" table exists in LanceDB by verifying the table names.
# 2. If the "embeddings" table is absent:
#     - Log a message indicating that the "embeddings" table is being created.
#     - Create a new table named "embeddings" using the predefined schema, which includes:
#         - `text_id` for unique document identifiers,
#         - `vector` for storing the embedding vectors, and
#         - `original_text` to hold the raw text of each document.
# 3. If the "embeddings" table already exists:
#     - Log a message confirming that the "embeddings" table is being opened.
#     - Open the existing "embeddings" table for use.
# ```

# Check if the "embeddings" table exists in LanceDB
if "embeddings" not in db:
    # Create a new table named "embeddings" with the specified schema
    collection = db.create_table("embeddings", schema=schema)
    logging.info("Created table 'embeddings'")
else:
    # Open the existing "embeddings" table
    collection = db["embeddings"]
    logging.info("Opened table 'embeddings'")

# Create an index for vector search (only if table is not empty)
try:
    # **TODO 3**: Create an index on the `vector` field to enhance the speed of vector-based searches, which are crucial for efficiently retrieving similar documents. Perform this step only if the table contains data, ensuring optimized retrieval for LangChain’s pipeline.
    # ```plaintext
    # pseudocode:
    # 1. Check if the "embeddings" table has any data by measuring its length:
    #     - If the table contains data:
    #         a. Create an index on the `vector` field, which stores document embeddings, to speed up similarity searches.
    #         b. Use the "IVF_FLAT" index type and specify 10 partitions for effective query performance.
    #     - If the table is empty:
    #         a. Log a message indicating that index creation is skipped because there is no data.
    # 2. Handle any exceptions during index creation, and if an error occurs, log a warning message with details about the issue.
    # ```
    if len(collection) > 0:
        # Create an index on the `vector` field for efficient similarity searches
        collection.create_index("vector", "IVF_FLAT", 10)
    else:
        logging.info("Skipping index creation as the table is empty")
except Exception as e:
    logging.warning(f"Could not create index: {str(e)}")
