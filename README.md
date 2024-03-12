# Skynet
Gen Ai model for manual document search and synthesis
Gautam Naik (naikgautam234@gmail.com) | +1 469-990-0287
To understand the model properly we can divide it into 2 parts.
1.	User Interface:
•	We have used streamlit library for user interface. 
•	We have used “st.file_uploader” for uploading pdf file as it can handle file of the size 200MB.
•	We have chat history option to optimize the model as well as save time.
•	Future Changes:
•	Can add multiple chat option for user to use multiple PDFs at same time.
•	Have a proper structured page for Chat History.
•	Picking points from Rajeshwari Ganesan’s talk about adding a relevance score and page numbers to show genuinity or to show that the model is not hallucinating.

2.	Model:
•	We have used OpenAI’s "text-embedding-ada-002" model for embedding the tokens.
•	Used "gpt-3.5-turbo-0125" as it can handle upto 4k tokens.
•	We have used “ChromaDB” for vector database storage.
•	To connect this model with User Interface we have used Langchain library.
•	User can upload a PDF locally also in case they want to. 
•	Used “Similarity search” for selecting retrieval context. 
Future Changes:
•	We can use gpt-4 instead of gpt-3.5 to uplift performance.
•	Can also introduce image generator.

  
