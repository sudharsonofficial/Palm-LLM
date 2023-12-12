# Import necessary libraries
from langchain.llms import GooglePalm
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv, find_dotenv
from langchain.docstore.document import Document
import google.generativeai as palm
import os 

# Function to load LangChain Language Model (LLM)
def load_llm():
    # Load environment variables from .env file
    load_dotenv(find_dotenv())
    # Get API key from environment variable
    api_key = os.environ['GOOGLE_API_KEY']  # put your API key here
    # Configure Google Palm API with the obtained key
    palm.configure(api_key=api_key)
    # Create an instance of GooglePalm LLM
    llm = GooglePalm()
    llm.temperature = 0.1
    # Load Question-Answering chain with the specified LLM type
    chain = load_qa_chain(llm, chain_type="stuff")
    return chain

# Function to query LangChain Language Model (LLM)
def query_llm(chain, query, docsearch):
    # Use document similarity search to find relevant passages
    docs = docsearch.similarity_search(query)
    relevant = docs[0].page_content

    # Template for generating a structured input for the LLM
    template = f"""
        You are a helpful and informative bot that answers questions using text from the passage included below. 
        Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. 
        However, you are talking to a non-technical audience, so be sure to break down complicated concepts and 
        strike a friendly and conversational tone. 
        Answer the questions elaborately in 200 to 300 words. 
        Do not answer in a short manner. 
        Answer the question only from the Passage given. Don't provide any external answers. 
        If the answer is not found in the passage, answer with "The question is not relevant to the document."
        Answer only from the passage. Do not use an external knowledge base. 
        QUESTION: '{query}'
        PASSAGE: '{relevant}'

        ANSWER:
    """
    
    # Run the LangChain LLM with the structured input
    res = chain.run(input_documents=[Document(page_content=template)], question=query).strip()
    return res
