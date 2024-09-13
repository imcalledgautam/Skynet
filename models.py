from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from loguru import logger
from getpass import getpass
import os
from utils import load_env_vars

class App:
    def _init_(self) -> None:
        load_env_vars()
        self.persist_directory = "db"
        self.embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        self.vectordb = None  # Initialize the vector database attribute
        self.llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0125")
        self.qa_chain = load_qa_chain(self.llm, chain_type="stuff")
        if not hasattr(st.session_state, 'chat_history'):
            st.session_state.chat_history = []
        self.version_description = "At SKYNET, we are revolutionizing the way industrial manufacturers access and utilize information..."

    def new_chat(self, chat_entry):
        st.session_state.chat_history.append(chat_entry)

    def clear_chat(self):
        st.session_state.chat_history.clear()

    def display_chat_history(self):
        if st.sidebar.button("Clear History"):
            self.clear_chat()
        st.sidebar.write(st.session_state.chat_history)

    def display_version_button(self):
        if st.sidebar.button("About us"):
            self.display_about_us_page()

    def display_about_us_page(self):
        st.sidebar.write(self.version_description)