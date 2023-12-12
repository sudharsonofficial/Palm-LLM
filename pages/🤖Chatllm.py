# Import necessary libraries
import streamlit as st
from llm2 import palmllm
from Pinecone import load_pine, pre_doc

# Set the title of the Streamlit app
st.title("Chat with Palm 🤖")

# Initialize chat history
if "messagespalm" not in st.session_state:
    st.session_state.messagespalm = []

# Display chat messages from history on app rerun
for message in st.session_state.messagespalm:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if query := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(query)
    # Add user message to chat history
    st.session_state.messagespalm.append({"role": "user", "content": query})
    # Retrieve the response from palm
    response = palmllm(query)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messagespalm.append({"role": "assistant", "content": response})