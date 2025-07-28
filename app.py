# sqlite fix for Chroma
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import concurrent.futures
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from task_doer_agent_new import run_task

st.title("🔁 Chroma + OpenAI Task Agent")

# --- API Key ---
openai_api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("❌ OpenAI API key not found.")
    st.stop()

# --- Embedding Model ---
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# --- Chroma DB Setup ---
persist_dir = "chroma_db"
db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

# --- UI ---
task = st.text_area("Enter your task:")

def run_in_thread(task):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(run_task, task)
        return future.result(timeout=300)

if st.button("Run Task"):
    if task.strip():
        with st.spinner("🔍 Searching Chroma..."):
            try:
                results = db.similarity_search(task, k=1)
                if results and results[0].metadata.get("user_task", "").lower() == task.lower():
                    st.subheader("🧠 Answer from ChromaDB")
                    st.write(results[0].page_content)
                else:
                    st.info("🤖 No exact match in memory. Asking OpenAI...")
                    result = run_in_thread(task)

                    if result:
                        st.subheader("✅ Result from OpenAI")
                        st.write(result)

                        # Save to Chroma
                        doc = Document(
                            page_content=result,
                            metadata={"source": "openai", "user_task": task}
                        )
                        db.add_documents([doc])
                        db.persist()
                        st.success("💾 Stored in Chroma for future use.")
                    else:
                        st.warning("⚠️ No result returned from OpenAI.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    else:
        st.warning("Please enter a task.")

