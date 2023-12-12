# Import necessary libraries
import streamlit as st
from llm import load_llm, query_llm
from Pinecone import load_pine, pre_doc

# Set the title of the Streamlit app
st.title("Q Palm 🤖")

# Sidebar for file upload
with st.sidebar:
    st.header("FILE UPLOAD")
    pdf = st.file_uploader("Supported document: PDF")
    st.info("Please submit your PDF document here. Kindly be aware that the processing time will increase with the document's size. Pose queries related to the document's content only for further assistance", icon='💡')
    
    # Clear cache if PDF is not uploaded
    if pdf is None:
        st.cache_resource.clear()
    else:
        # Cache resource for document processing
        @st.cache_resource(show_spinner=False)
        def process_document(pdf_path):
            with st.status("Analysing Your Document", state="running", expanded=False) as status:
                st.write("Preprocessing your Document...")
                text = pre_doc(pdf_path)
                st.write("Creating a Vector Database...")
                doc_search = load_pine(text)
                st.write("Loading the LLM")
                chain = load_llm()
                status.update(label="Completed", state="complete", expanded=False)
            return doc_search, chain

        # Process the uploaded document
        doc_search, chain = process_document(f'Books\{pdf.name}')

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if pdf is None:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if pdf is not None:
    if query := st.chat_input("What is up?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(query)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})
        response = query_llm(chain, query, doc_search)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})