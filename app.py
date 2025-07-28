import streamlit as st
import concurrent.futures
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
import os
from task_doer_agent_new import run_task

st.title("🧠 TaskDoer Agent")
task = st.text_area("Enter your task:")


# Load OpenAI key securely
openai_api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("❌ OpenAI API key not found in secrets or environment.")
    st.stop()

# Set up embedding model
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)


# Load Chroma DB (persistent)
persist_dir = "chroma_db"
if not os.path.exists(persist_dir):
    # Optional: First-time setup if DB doesn't exist
    loader = TextLoader("my_docs.txt")
    docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    db = Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)
    db.persist()
else:
    db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

# Search
if st.button("Run Task"):
    if task.strip():
        with st.spinner("🔍 Searching..."):
            try:
                results = db.similarity_search(task, k=3)
                if results:
                    st.subheader("✅ Top Results:")
                    for i, doc in enumerate(results):
                        st.markdown(f"**Chunk {i+1}**:\n{doc.page_content}")
                else:
                    st.info("No relevant content found.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    else:
        st.warning("Please enter a task.")

def run_in_thread(task):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(run_task, task)
        return future.result(timeout=300)  # Adjust timeout if needed

#if st.button("Run Task"):
 #   if task.strip():
  #      with st.spinner("Thinking..."):
   #         try:
    #            result = run_in_thread(task)
     #           st.subheader("✅ Result")
      #          st.write(result)
       #     except concurrent.futures.TimeoutError:
        #        st.error("⏱️ Task took too long and timed out.")
    #else:
     #   st.warning("Please enter a task.")




