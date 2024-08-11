import os
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.docstore.document import Document
from langchain.vectorstores.faiss import FAISS
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Define system template
system_template = """Use the following pieces of context to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
"""

# Load environment variables
# load_dotenv()

# Define message templates
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]

os.environ["GOOGLE_API_KEY"] = "AIzaSyCMtgjCDQl7iBJsLa2iGTH0KBmsIw3UGFo"

# Create chat prompt template
prompt = ChatPromptTemplate.from_messages(messages)

# Define constants
ABS_PATH = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(ABS_PATH, "db")

# Function to process file and create chunks based on new lines
def process_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.read().split('\n')

    # Create Document objects from each line
    docs = [Document(page_content=line.strip()) for line in lines if line.strip()]

    # Initialize embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Create a FAISS vector database from the documents
    vectordb = FAISS.from_documents(documents=docs, embedding=embeddings)
    vectordb.save_local(DB_DIR)

    return vectordb

# Function to process query
def process_query(vectordb, query, chat_history):
    # Create a retriever from the FAISS vector database
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    # Use a Llama3-70b model
    llm = ChatGroq(
        api_key="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",
        temperature=0,
        model="llama3-70b-8192",
    )

    # Create a ConversationalRetrievalChain with a StuffedDocumentChain
    chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type="stuff",
        verbose=True,
    )

    # Format chat history to a list of tuples
    formatted_chat_history = [(item['question'], item['answer']) for item in chat_history]

    # Run the prompt and return the response
    response = chain({"question": query, "chat_history": formatted_chat_history})

    return response

def main():
    # Set the title and subtitle of the app
    st.title("🦜🔗 Chat With Files")
    st.header("Ask Questions")
    query = st.text_input("Ask a question (query/prompt)")

    # Sidebar for file input
    with st.sidebar:
        st.header("Upload and Process File")
        uploaded_file = st.file_uploader("Upload a text file")
        if uploaded_file and st.button("Process File"):
            file_path = os.path.join(DB_DIR, uploaded_file.name)
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            vectordb = process_file(file_path)
            st.session_state['vectordb'] = vectordb
            st.success("File processed and data embedded successfully.")
            st.session_state['file_processed'] = True

    # Main area for query input and responses
    if 'file_processed' in st.session_state and st.session_state['file_processed']:
        if st.button("Submit Query", key="query_submit"):
            chat_history = st.session_state.get('chat_history', [])
            response = process_query(st.session_state['vectordb'], query, chat_history)
            st.write(response["answer"])
            chat_history.append({"question": query, "answer": response["answer"]})
            st.session_state['chat_history'] = chat_history

if __name__ == "__main__":
    main()
