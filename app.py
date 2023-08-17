import streamlit as st
from PyPDF2 import PdfReader
# from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
#Sidebar contents
import pickle 
from dotenv import load_dotenv
import os


with st.sidebar:
    st.title('LLM Chat App')
    st.markdown('''
    ##ABOUT

    This is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [Langchain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model




    ''')
    # add_vertical_space(5)
    st.write("Made by ruxiiiii")
store_name = ''
chunks = []

def main():
    st.header('Chat with any PDF')
    
    load_dotenv()


    pdf = st.file_uploader("Upload your PDF", type = 'PDF')
    
 
 
    if pdf is not None:
        pdf_reader  = PdfReader(pdf)
        # st.write(pdf_reader)
        st.write(pdf.name)

        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()
        

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function = len
        )

        global chunks
        chunks = text_splitter.split_text(text=text)
        st.write(chunks)
        global store_name
        store_name = pdf.name[:-4]
        
    #embeddings
    
    
    if os.path.exists(f'{store_name}.pkl'):
        
        with open(f'{store_name}.pkl', 'rb') as f:
             VectorStore = pickle.load(f)
        st.write('Embeddings loaded from disk')

    else:
        
        global embeddings
        embeddings = OpenAIEmbeddings()
        VectorStore = FAISS.from_texts(chunks, embedding = embeddings)
        
        with open(f'{store_name}.pkl','wb') as f:
            pickle.dump(VectorStore,f)




    

if __name__ == '__main__':
    main()
