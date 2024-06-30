# Chat with your PDF Document!

This project put together by Manogya allows users to upload PDF documents, extract text from them, and interact with the content using a conversational AI model.

## Features

- Upload a single or multiple PDF files.
- Extract text from the uploaded PDFs.
- Split the extracted text into manageable chunks.
- Store text embeddings locally using FAISS.
- Ask questions related to the uploaded PDF content and receive detailed answers.

## Requirements

- Python 3.x
- Required libraries: 
  - os
  - dotenv
  - pickle
  - google.generativeai
  - streamlit
  - PyPDF2
  - pathlib
  - langchain_google_genai
  - langchain
  - langchain_community

## Installation

1. Clone the repository:

    ```bash
    https://github.com/manogyaguragai/Talk-to-PDF.git
    ```

2. Install the required libraries:

    ```bash
    pip install -r requirements.txt
    ```

3. Create a `.env` file in the root directory and add your Google API key:

    ```env
    GOOGLE_API_KEY=your_google_api_key
    ```

## Usage

1. Run the Streamlit application:

    ```bash
    streamlit run talk_to_pdf.py
    ```

2. In the Streamlit interface:
    - Upload your PDF files.
    - Click the "Submit" button to process the PDFs.
    - Ask questions related to the content of the uploaded PDFs in the provided input box.

## Code Overview

- `get_pdf_text(pdf_docs)`: Extracts text from all uploaded PDF files.
- `get_text_chunks(text)`: Splits the extracted text into chunks.
- `get_vector_store(text_chunks)`: Creates a local vector store for text embeddings.
- `get_conversation_chain(retr)`: Creates a conversational chain with a predefined prompt and selected model.
- `user_input(user_question)`: Handles user input, performs similarity search, and retrieves answers.
- `main()`: Main function to run the Streamlit application.


### Points to Note

- It is recommended to fact-check the answers provided by this model.
- The information may sometimes be incomplete due to limited output size.


