import os

import streamlit as st
from dotenv import find_dotenv, load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from loguru import logger
from getpass import getpass
import base64

_ = load_dotenv(find_dotenv())

# os.getenv('OPENAI_API_KEY')
OPENAI_API_KEY = getpass()
os.environ["OPENAI_API_KEY"] = "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"

st.title("SkyNet\n -sky is the limit")
st.header("How can I help you today?")

class App:
    def __init__(self) -> None:
        self.persist_directory = "db"
        self.embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        self.vectordb = None  # Initialize the vector database attribute

        self.llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0125")
        self.qa_chain = load_qa_chain(self.llm, chain_type="stuff")

        # Initialize session_state for storing chat history
        if not hasattr(st.session_state, 'chat_history'):
            st.session_state.chat_history = []

        # Version information
        self.version_description = "At SKYNET, we are revolutionizing the way industrial manufacturers access and utilize information. Our cutting-edge generative AI technology is designed to streamline document retrieval, enabling businesses to extract valuable insights from their data faster than ever before\n Our AI-powered system quickly scans and summarizes key information from manuals, technical documents, and other data sources, delivering concise and actionable insights in real-time. Say goodbye to the tedious task of manually sifting through vast amounts of information\n Empower your workforce with intelligent self-help tools that provide instant access to relevant information, reducing downtime and increasing productivity. SKYNET's AI-driven knowledge base ensures that your employees always have the answers they need, when they need them\n By analyzing patterns and anomalies in your data, SKYNET can identify potential trouble areas, bottlenecks, and inefficiencies in your manufacturing processes. Armed with these insights, you can take proactive measures to optimize operations, reduce costs, and improve overall efficiency\n In today's fast-paced industrial landscape, the ability to quickly extract value from data can mean the difference between success and failure. With SKYNET, you gain a powerful ally that harnesses the full potential of AI, giving you a competitive edge in an increasingly data-driven world"



    def display_version_button(self):
        # Create an empty space on the right to position the button in the top right corner
        st.markdown("<div style='position: absolute; top: 60px; right: 60px;'></div>", unsafe_allow_html=True)

        # Display the version button in the empty space
        version_button = st.sidebar.button("About us", key="version_button")
        return version_button
    
    def display_about_us_page(self):
        st.title("About Us")
        st.markdown(self.version_description)


    def load_and_split(self, path: str):
        logger.info(f"Loading {path}")
        loader = PyPDFLoader(path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(docs)

        return texts

    def process_pdf(self, pdf_content):
        self.vectordb = Chroma.from_texts(
            texts=[c.page_content for c in pdf_content],
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
        )
        self.vectordb.persist()

    def process_uploaded_pdf(self, uploaded_pdf):
        temp_file = "./temp.pdf"
        with open(temp_file, "wb") as file:
            file.write(uploaded_pdf.getvalue())
            file_name = uploaded_pdf.name
            logger.info(f"Uploaded {file_name}")

        pdf_content = self.load_and_split(path=temp_file)
        self.process_pdf(pdf_content)

    def pre_specified_pdfs(self):
        # Specify pre-existing PDFs here
        pdf_paths = ["D:/MSBA/SKYNET/1034384-2P_MILLPWR_G2_Users_Manual.pdf"]
        pdf_content = []
        for pdf_path in pdf_paths:
            pdf_content.extend(self.load_and_split(pdf_path))
        self.process_pdf(pdf_content)

    def new_chat(self, chat_entry):
        # Add the new chat entry to the chat history
        st.session_state.chat_history.append(chat_entry)

    def display_chat_history(self):
        st.sidebar.header("Chat History")
        # Reverse the chat history list to display the latest entry first
        reversed_chat_history = reversed(st.session_state.chat_history)


        # Display each chat entry in the sidebar
        for i, chat_entry in enumerate(reversed_chat_history):
            st.sidebar.subheader(f"Chat Entry {len(st.session_state.chat_history) - i}")
            st.sidebar.write(chat_entry)
            st.sidebar.markdown("---")  # Add a separator between chat entries


    def clear_chat(self):
        # Clear the chat history
        st.session_state.chat_history = []
    
    def __call__(self):
        #
        # Store Pre-specified PDFs in DB

        self.pre_specified_pdfs()

        #
        # Store PDF file in DB (if uploaded)
        #
        uploaded_pdf = st.file_uploader("Upload a PDF")
        if uploaded_pdf:
            self.process_uploaded_pdf(uploaded_pdf)

        #
        # Prompt question, language, and optimization options
        #
        question = st.text_input("Ask a question")
        if question and self.vectordb:
            sim_methods = ["similarity_search"]
            sim_method = sim_methods[0]
            logger.info(f"Similarity method: {sim_method}")

            m = self.vectordb.similarity_search if sim_method == sim_methods[0] else None

            q_rs = m(
                question,
                k=1
                # num_docs
                # fetch_k=6,
                # lambda_mult=1
            )

            #
            # Start chain for concrete answer
            #
            question = f"{question}"
            answer = self.qa_chain.run(input_documents=q_rs, question=question)
            st.write(answer)

            #
            # Update chat history with the current question and answer
            #
            chat_entry = f"{question} ---> \n{answer}"
            self.new_chat(chat_entry)

            #
            # Display chat history in the sidebar
            #
        
        if st.sidebar.button("Clear History"):
            self.clear_chat()

        version_button = self.display_version_button()

        # Check if the "About us" button is clicked
        if version_button:
            self.display_about_us_page()

        self.display_chat_history()

        #self.display_version_button()

        # Clear Chat Button
        #
        
if __name__ == "__main__":
    app = App()
    app()
