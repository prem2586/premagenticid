import streamlit as st
import concurrent.futures
from task_doer_agent_new import run_task

st.title("🧠 TaskDoer Agent")
task = st.text_area("Enter your task:")

def run_in_thread(task):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(run_task, task)
        return future.result(timeout=300)  # Adjust timeout if needed

if st.button("Run Task"):
    if task.strip():
        with st.spinner("Thinking..."):
            try:
                result = run_in_thread(task)
                st.subheader("✅ Result")
                st.write(result)
            except concurrent.futures.TimeoutError:
                st.error("⏱️ Task took too long and timed out.")
    else:
        st.warning("Please enter a task.")




