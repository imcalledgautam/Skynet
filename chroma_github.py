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

_ = load_dotenv(find_dotenv())

# os.getenv('OPENAI_API_KEY')
OPENAI_API_KEY = getpass()
os.environ["OPENAI_API_KEY"] = "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"


page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
background-color: #e5e5f7;
opacity: 0.8;
background-image: radial-gradient(circle at center center, #444cf7, #e5e5f7), repeating-radial-gradient(circle at center center, #444cf7, #444cf7, 10px, transparent 20px, transparent 10px);
background-blend-mode: multiply;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

st.title("Ask Skynet")

class App:
    def __init__(self) -> None:
        self.persist_directory = "db"
        self.embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

        self.llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0125")
        self.qa_chain = load_qa_chain(self.llm, chain_type="stuff")

        self.lang_list = [
            "English",
            "German",
            "Simplified Chinese",
            "Traditional Chinese",
        ]

    def load_and_split(self, path: str):
        logger.info(f"Loading {path}")
        loader = PyPDFLoader(path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(docs)

        return texts

    def __call__(self):
        # pdf_options = ["Unstructured PDF", "Normal PDF"]
        # pdf_op = st.radio("Select PDF type", pdf_options, index=0)

        #
        # Store PDF file in DB
        #
        uploaded_pdf = st.file_uploader("Upload a PDF")
        if uploaded_pdf:
            temp_file = "./temp.pdf"
            with open(temp_file, "wb") as file:
                file.write(uploaded_pdf.getvalue())
                file_name = uploaded_pdf.name
                logger.info(f"Uploaded {file_name}")

            pdf_content = self.load_and_split(
                path=temp_file
                # , is_unstructured=pdf_op == pdf_options[0]
            )
            vectordb = Chroma.from_texts(
                texts=[c.page_content for c in pdf_content],
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
            )
            vectordb.persist()

            os.remove(temp_file)
            uploaded_pdf = None

            #
            # Prompt question, language and optimisation options
            #
            question = st.text_input("Ask a question")
            cols = st.columns(2)
            # src_lang = cols[0].selectbox("Source Language", self.lang_list)
            # dist_lang = cols[1].selectbox("Destination Language", self.lang_list)
            cols = st.columns(2)
            # num_docs = cols[0].number_input(
            #     "Number of documents to retrieve", min_value=1, max_value=10, value=4
            # )
            cols[1].write("")
            cols[1].write("")
            # is_opt = cols[1].checkbox("Optimize result", value=False)
            # optimising = "necessary" if is_opt else "unnecessary"

            if question:
                sim_methods = ["similarity_search"]
                sim_method = sim_methods[0]
                logger.info(f"Similarity method: {sim_method}")

                #sim_methods = ["similarity_search"]
                #sim_method = st.radio("Similarity method", sim_methods, index=0)
                #logger.info(f"Similarity method: {sim_method}")

                #
                # Retrieve the most similar documents.
                # For max_marginal_relevance_search, the documents are far-apart from each other.
                #
                m = vectordb.similarity_search if sim_method == sim_methods[0] else None

                #(
                #    vectordb.similarity_search
                #    if sim_method == sim_methods[0]
                #    else vectordb.max_marginal_relevance_search
                #)
                q_rs = m(
                    question,
                    k=1
                    # num_docs
                    # fetch_k=6,
                    # lambda_mult=1
                )

                #
                # Start chain for concerate answer
                #
                question = f"{question}(answer)"
                answer = self.qa_chain.run(input_documents=q_rs, question=question)
                st.write(answer)


if __name__ == "__main__":
    app = App()
    app()
