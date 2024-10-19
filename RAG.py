import os
import uuid
import streamlit as st
import tempfile
from datetime import datetime
import glob
import openai
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools.retriever import create_retriever_tool
from langchain.schema.messages import HumanMessage, AIMessage, ToolMessage
from langchain.schema.runnable import RunnableConfig
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader


openai.api_key = ""

os.environ["OPENAI_API_KEY"] = openai.api_key
tmp_directory = tempfile.mkdtemp()

class HelpBot:

    def __init__(self):
        st.title("Help Bot")
        st.write("Ask anything about the Uploaded Document")
        self.last_db_history = None
        self.uploaded_files = []

    def load_docs(self):
        # Load documents from the given directory.
        loader = DirectoryLoader(tmp_directory, glob="**/*.txt", loader_cls=TextLoader)
        documents = loader.load()

        return documents

    def split_docs(self, documents, chunk_size=500, chunk_overlap=20):
        # Split the documents into chunks.

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        splits = text_splitter.split_documents(documents)

        return splits

    def start_app(self):
        # Load all necessary models and data once the server starts.

        print(f"Last DB update: {self.last_db_history}")

        # Load documents from the uploaded files
        documents = self.load_docs()

        if not documents:
            st.warning("No documents loaded. Please upload files first.")
            return None, None

        embeddings = HuggingFaceEmbeddings()
        splits = self.split_docs(documents)
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        retriever = vectorstore.as_retriever()

        llm = ChatOpenAI(model="gpt-3.5-turbo", streaming=True)

        return retriever, llm

    def start_chat(self):

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if user_prompt := st.chat_input("Ask me anything about the uploaded document"):
            st.session_state.messages.append({"role": "user", "content": user_prompt})
            with st.chat_message("user"):
                st.markdown(user_prompt)

            with st.chat_message("assistant"):
                response = self.get_answer(user_prompt)
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

    def get_answer(self, query: str):
        retriever, llm = self.start_app()
        if retriever is None or llm is None:
            return "No documents loaded or database connection issue. Please try uploading files again."

        memory = MemorySaver()
        thread_id = str(uuid.uuid4())
        config = RunnableConfig(configurable={"thread_id": thread_id})

        retriever_tool = create_retriever_tool(
            retriever,
            "search_documents",
            "Searches and returns documents relevant to the query."
        )

        tools = [retriever_tool]
        agent_executor = create_react_agent(llm, tools, checkpointer=memory)

        # Convert session messages to the format expected by the agent
        chat_history = [
            HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"])
            for msg in st.session_state.messages
        ]

        # Add the current query to the chat history
        chat_history.append(HumanMessage(content=query))

        # Stream the agent's response
        response_parts = []
        for s in agent_executor.stream(
                {"messages": chat_history},
                config=config
        ):
            if isinstance(s, dict):
                if 'agent' in s and 'messages' in s['agent']:
                    for message in s['agent']['messages']:
                        if isinstance(message, AIMessage) and message.content:
                            response_parts.append(message.content)
                            # Optionally, update the Streamlit UI in real-time
                            st.write(message.content)
                elif 'action' in s and 'messages' in s['action']:
                    for message in s['action']['messages']:
                        if isinstance(message, ToolMessage) and message.content:
                            # You might want to handle tool messages differently
                            st.write(f"Tool used: {message.name}")
                            st.write(message.content)

        # Combine all parts of the response
        full_response = " ".join(response_parts)

        return full_response


# Initialize the HelpBot
helper = HelpBot()

context = st.sidebar.radio("Which knowledge base do you want to use?",
                           ["Already uploaded", "Upload new one"])

# File upload logic
if context == "Upload new one":
    uploaded_files = st.sidebar.file_uploader("choose a text file", accept_multiple_files=True)

    if uploaded_files:
        # Clear temp directory
        for file in os.listdir(tmp_directory):
            os.remove(os.path.join(tmp_directory, file))

        # Save uploaded files to the temporary directory
        for file in uploaded_files:
            with open(os.path.join(tmp_directory, file.name), "wb") as f:
                f.write(file.getvalue())

        helper.last_db_history = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        st.success(f"{len(uploaded_files)} files uploaded successfully!")

# Display current files in the knowledge base
current_files = os.listdir(tmp_directory)
if context == "Already uploaded":
    st.sidebar.write("Current Knowledge Base")
    if current_files:
        st.sidebar.write(current_files)
    else:
        st.sidebar.write("**No files uploaded**")

    if helper.last_db_history:
        st.sidebar.write(f"Last updated: {helper.last_db_history}")
else:
    if current_files:
        st.sidebar.write("Files ready to be processed:")
        st.sidebar.write(current_files)
    else:
        st.sidebar.write("No files selected for upload")

if context == "Upload new one" and uploaded_files is not None and len(uploaded_files):
    helper.last_db_history = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

helper.start_chat()