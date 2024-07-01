import os
from dotenv import load_dotenv
import pickle

import google.generativeai as genai

import streamlit as st

import PyPDF2

import pathlib

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain.chains.question_answering import load_qa_chain 
from langchain.prompts import PromptTemplate 
load_dotenv()

st.set_page_config(page_title = "Talk To PDF")

genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))

# Function to iterate over all PDFs and return the extracted text from all pages 
def get_pdf_text(pdf_docs):
    text = ""
    # iterate over all pdf files uploaded
    for pdf in pdf_docs:
        pdf_reader = PyPDF2.PdfReader(pdf)
        # iterate over all pages in a pdf
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to divide the text into chunks
def get_text_chunks(text):
    # create an object of RecursiveCharacterTextSplitter with specific chunk size and overlap size
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
    # now split the text we have using object created
    chunks = text_splitter.split_text(text)
    return chunks

# Function that will create a place locally to store the embeddings as a vector quantity 
def get_vector_store(text_chunks, allow_dangerous_deserialization=True):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/text-embedding-004") 
    vector_store = FAISS.from_texts(text_chunks,embeddings) 
    #vector_store = FAISS.from_documents(text_chunks, embeddings)
    vector_store.save_local("faiss_index")  
    return vector_store

# FUnction that makes a conversation chain with a predefined prompt, a selected model, and user input
def get_conversation_chain(retr):
    # define the prompt
    prompt_template = """
    You are a capable AI model that answers the question as detailed and effectively as possible from the provided context.\n
    Your answer should be as close to the provided context as possible.\n
    Give as many results as specified by the user by focusing on quantifying words. If user does not specify, give a concise answer.\n
    Be polite to user queries and respond properly to user greetings. \n
    If the user asks to describe the file type, make sure to do it relevantly and descriptively.\n
    Make sure to provide all the details, if the answer is not in provided context just tell the user that you could not find the relevant answer in the given context, don't provide the wrong answer.\n\n

    Context: {context}\n
    Question: {question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model = "gemini-1.5-pro", temperatue = 0.0) # create object of gemini-pro

    prompt = PromptTemplate(template = prompt_template, input_variables= ["context","question"])

    #vec_store = get_vector_store()
    chain = load_qa_chain(model, chain_type="stuff", prompt = prompt)

    return chain

# Function to take in the user's question and perform similarity search
def user_input(user_question):
    # user_question is the input question
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/text-embedding-004")
    # load the local faiss db
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = new_db.as_retriever(search_type = 'mmr', search_kwargs={"k": 3})
    # using similarity search, get the answer based on the input
    # docs = new_db.similarity_search(user_question)

    chain = get_conversation_chain(retriever)   
    docs = retriever.invoke(user_question)

    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    #st.write(response["output_text"])
    st.chat_message('user').write(user_question)
    st.chat_message('ai').write(response["output_text"])



# Main functio to tie it all up
def main():  
    st.header("Chat with your PDF Document!")
    st.markdown("<h5 style='color: grey;'>Works with PDF files only</h1>", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("<h2 style='color: red;'>Instruction Menu</h1>", unsafe_allow_html=True)

        st.markdown("<h4 style='color: grey;'>1. Upload your PDF Files using the 'Browse Files' button.</h4>", unsafe_allow_html=True)
        pdf_docs = st.file_uploader("", type = "pdf", accept_multiple_files=True)
        
        if pdf_docs:
            file_name = pdf_docs[0].name

            if file_name.endswith(".pdf"):
                if st.button("Submit", type = "primary"):
                    with st.spinner("Processing..."):
                        raw_text = get_pdf_text(pdf_docs)
                        text_chunks = get_text_chunks(raw_text)
                        vec_store = get_vector_store(text_chunks)
                        # get_conversation_chain()
                        st.success("Done")
            else:
             st.error("Please upload a valid PDF file.")

        st.markdown("<h4 style='color: grey;'>2. Click on the Submit Button</h4>", unsafe_allow_html=True)
        st.markdown("<h4 style='color: grey;'>3. Ask a Question related to your document and get an answer.</h4>", unsafe_allow_html=True)
        st.markdown("<h4 style='color: grey;'>It's just that easy!</h4>", unsafe_allow_html=True)

        # st.markdown("<h2 style='color: red;'>Points to Note</h1>", unsafe_allow_html=True)
        # st.markdown("<h4 style='color: grey;'>1. It is recommended that you fact-check the answers provided by this model.</h4>", unsafe_allow_html=True)
        # st.markdown("<h4 style='color: grey;'>2. The information may sometimes be incomplete due to limited output size.</h4>", unsafe_allow_html=True)

    user_question = st.chat_input("Ask a Question:")

    if user_question:
        user_input(user_question)

    
if __name__ == "__main__":
    main()


