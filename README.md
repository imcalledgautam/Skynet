# Skynet PDF Chatbot
Gen AI model for manual document search and synthesis
Gautam Naik (naikgautam234@gmail.com) | +1 469-990-0287

#Overview

Skynet PDF Chatbot is an AI-powered assistant that allows users to upload PDF documents and query them using natural language. The chatbot leverages OpenAI's GPT-3.5-turbo, Chroma vector database, and LangChain to retrieve and generate relevant answers based on document content.

#Features
📄 PDF Upload & Processing – Supports document uploads for AI-powered querying.

🔍 Semantic Search – Uses OpenAI embeddings to find the most relevant content.

🤖 AI-Powered Responses – Retrieves answers based on document context.

🗂 Chat History – Stores and displays past user queries.

🖥 Streamlit UI – Interactive web interface for an easy user experience.

#Installation & Setup

1. Clone the Repository
   git clone https://github.com/imcalledgautam/Skynet
   cd skynet-pdf-chatbot
2. Usage
Run the Application
streamlit run test.py

#Functionality Breakdown
1. PDF Processing

The app uses PyPDFLoader from LangChain to extract text from uploaded PDFs.
It splits text into manageable chunks using RecursiveCharacterTextSplitter.

2. Vector Database

Uses Chroma as a persistent vector store.
Converts PDF text into embeddings using OpenAI's text-embedding-ada-002 model.

3. Question Answering

Implements ChatOpenAI from LangChain for query processing.
Retrieves relevant document sections using similarity search.

4. User Interaction

Built using Streamlit for an interactive UI.
Users can upload PDFs, enter questions, and receive responses.
Includes a chat history feature to revisit previous queries.

#Known Issues & TODO
✅ Implemented:

-Basic PDF upload and querying.
-Semantic search using OpenAI embeddings.
-Streamlit UI with chat history.

#🔜 Upcoming Improvements:

-Allow users to select different similarity search methods.
-Multi-document querying.
-UI enhancements.
