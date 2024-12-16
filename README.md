# Prompt Tester LangChain

## Overview

Prompt Tester LangChain is an application designed to test and evaluate various AI-generated prompts. This tool helps developers and researchers to fine-tune their prompts for better performance and accuracy.

The application uses RAG (Retrieval Augmented Generation) to query and existing data set and leveage ChatGPT for forumlating responses to prompts.

## Features

- Test AI-generated prompts
- Evaluate prompt performance
- Retrieval Augmented Generation
- Langchain

## Setup Instructions

### Prerequisites

- Python 3.12 (see .tool-versions for recommended version)
- pip (Python package installer)

### Installation

1. Clone the repository:

2. Create a virtual environment for both applications (main_app & rag_service):

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt # For the main_app
pip install -r requirements_rag.txt # For the rag_services
```

## Starting the Servers

### Rag Service (backend)

1. Ensure you are in the `rag_services` directory and the virtual environment is activated.
2. Start the server:

```bash
uvicorn rag_service:app --reload
```

3. Open your web browser and navigate to `http://localhost:8000/docs` to access the application.

### Main Application (frontend)

1. Ensure you are in the `main_app` directory and the virtual environment is activated.
2. Start the server:

```bash
streamlit run prompt_tester.py
```

3. Open your web browser and navigate to `http://localhost:8501` to access the application.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
