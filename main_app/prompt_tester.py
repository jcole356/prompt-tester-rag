# prompt_tester.py
import streamlit as st
from prompt_quality_tester import PromptQualityTester, create_radar_chart
from rag_service_client import RAGServiceClient


def main():
    st.set_page_config(page_title="Prompt Quality Tester", layout="wide")
    st.title("Prompt Quality Tester")

    # Initialize RAG client
    rag_client = RAGServiceClient()

    # Check if the embedding count is already in session state; if not, fetch it
    if 'embedding_count' not in st.session_state:
        try:
            # Call the check_embeddings endpoint and store the count
            response = rag_client.check_embeddings()
            st.session_state['embedding_count'] = response['number_of_embeddings']
        except Exception as e:
            st.session_state['embedding_count'] = "Error fetching count"
            st.error(f"Failed to fetch embedding count: {str(e)}")
    
    # Sidebar for API key and instructions
    with st.sidebar:
        api_key = st.text_input("Enter OpenAI API Key:", type="password")

        # Document Embedding Section
        st.subheader("Document Embedding")

        if st.button("Embed All Documents", key="embed_button"):
            with st.spinner("Embedding all documents..."):
                try:
                    embed_result = rag_client.embed_all_documents()
                    st.success("All documents embedded successfully!", icon="✅")
                    # Update the embedding count after embedding
                    embedding_count_result = rag_client.check_embeddings()
                    st.session_state['embedding_count'] = embedding_count_result["number_of_embeddings"]
                except Exception as e:
                    st.error(f"Failed to embed documents: {str(e)}")

        if st.button("Clear Embeddings", key="clear_button"):
            with st.spinner("Clearing all embeddings..."):
                try:
                    clear_result = rag_client.clear_embeddings()
                    st.success("Embeddings cleared successfully!", icon="✅")
                    # Set the embedding count to 0 after clearing
                    st.session_state['embedding_count'] = 0
                except Exception as e:
                    st.error(f"Failed to clear embeddings: {str(e)}")

        # Display the updated embedding count after actions
        st.markdown("### Database Status")
        st.write(f"Number of embeddings: {st.session_state['embedding_count']}")


        st.markdown("### Task Types")
        st.write("""
        - Create a Tutorial
        - Explain Employee Benefits
        - Summarize Employee Handbook Section
        - Answer a Frequently Asked Question (FAQ)
        - Summarize Internal Memo
        - Describe a Job Role
        """)
        
        st.markdown("### Metrics Explanation")
        st.write("""
        - **Clarity**: Readability and comprehension
        - **Relevance**: Match with expected pattern
        - **Completeness**: Coverage of required elements
        - **Consistency**: Internal coherence
        - **Conciseness**: Information density
        """)

    if not api_key:
        st.warning("Please enter your OpenAI API key in the sidebar to continue.")
        return

    # Initialize the RAG client and tester
    rag_client = RAGServiceClient()
    tester = PromptQualityTester(api_key, rag_client)

    # Main interface
    col1, col2 = st.columns(2)

    with col1:
        task_type = st.selectbox("Select Task Type", [
            "Create a Tutorial",
            "Explain Employee Benefits",
            "Summarize Employee Handbook Section",
            "Answer a Frequently Asked Question (FAQ)",
            "Summarize Internal Memo",
            "Describe a Job Role",
        ])

        example_patterns = {
            "Create a Tutorial": "Step-by-step instructions with prerequisites and expected outcomes",
            "Explain Employee Benefits": "Provide a clear summary of employee benefits, including types of benefits, eligibility, and any prerequisites.",
            "Summarize Employee Handbook Section": "Summarize the key points in a section of the employee handbook, including rules, expectations, and guidelines for employees.",
            "Answer a Frequently Asked Question (FAQ)": "Provide a concise answer to a common employee question, ensuring clarity and accuracy of information.",
            "Summarize Internal Memo": "Summarize the main points of an internal memo, highlighting important updates or instructions for employees.",
            "Describe a Job Role": "Outline the key responsibilities, qualifications, and performance expectations for the job role described.",
        }


        prompt = st.text_area("Enter your prompt:", height=150)
        expected = st.text_area(
            "Enter expected response pattern:",
            value=example_patterns[task_type],
            height=100
        )

    if st.button("Test Prompt"):
        if prompt and expected:
            with st.spinner("Testing prompt..."):
                try:
                    # Retrieve enhanced prompt and LLM response using RAG client
                    result = rag_client.enhanced_retrieve(prompt, api_key)

                    # Display enhanced prompt
                    st.subheader("Enhanced Prompt Submitted to LLM:")
                    st.write(result['enhanced_prompt'])
                    
                    # Display LLM response
                    st.subheader("LLM Response:")
                    st.write(result['llm_response'])
                    
                    # Display information about documents used
                    st.subheader("Documents Used to Enhance Prompt:")
                    for doc in result["documents_used"]:
                        st.write(f"Text ID: {doc['text_id']}")
                        st.write(f"Snippet: {doc['snippet']}")
                        st.write(f"Score: {doc['score']}")
                        st.write("---")

                    # Perform quality testing on the enhanced prompt
                    metrics_result = tester.test_prompt(result['enhanced_prompt'], expected)

                    # Check for errors in metrics result
                    if "error" in metrics_result:
                        st.error(metrics_result["error"])
                    else:
                        # Display metrics and radar chart if no errors
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Quality Metrics")
                            st.metric("Overall Score", f"{metrics_result['metrics']['overall']:.2f}")
                            
                        with col2:
                            st.plotly_chart(create_radar_chart(metrics_result['metrics']))
                        
                        # Display detailed metrics
                        st.subheader("Detailed Metrics")
                        metrics_cols = st.columns(5)
                        for i, (metric, value) in enumerate(metrics_result['metrics'].items()):
                            if metric != 'overall':
                                metrics_cols[i].metric(metric.capitalize(), f"{value:.2f}")
                
                except Exception as e:
                    st.error(f"An error occurred during testing: {str(e)}")

if __name__ == "__main__":
    main()
