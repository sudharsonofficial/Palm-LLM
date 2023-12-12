# Import necessary libraries
from dotenv import load_dotenv, find_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GooglePalmEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
import os

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Get Pinecone API key from environment variable
papi_key = os.environ['PINE']

# Function to preprocess a PDF document and load it into Pinecone
def pre_doc(path):
    # Use PyPDFLoader to load the PDF document
    loader = PyPDFLoader(path)
    file = loader.load()
    
    # Use RecursiveCharacterTextSplitter to split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(file)
    
    return texts

# Function to load documents into Pinecone for similarity search
def load_pine(texts):
    # Get Pinecone API key from environment variable
    pine_api = os.environ['PINE']
    
    # Initialize Pinecone API
    pinecone.init(
        api_key=pine_api,
        environment='Specify Your Environment Name'
    )
    
    # Create Google Palm embeddings
    embeddings = GooglePalmEmbeddings()
    
    # Specify the index name
    index = 'Specify Your Index Name'
    
    # Check if the index already exists and delete it if it does
    indices = pinecone.list_indexes()
    if indices is not None:
        pinecone.delete_index(index)
    
    # Create a new index
    if index not in pinecone.list_indexes():
        pinecone.create_index(
            name=index,
            dimension=768,
            metric='cosine'
        )
    
    # Load documents into Pinecone
    docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index)
    
    return docsearch
