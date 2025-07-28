import os
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.llms import OpenAI
from langchain_experimental.tools import PythonREPLTool
from duckduckgo_search import DDGS
from dotenv import load_dotenv

load_dotenv()

# 1. LLM
llm = OpenAI(temperature=0.5)

# 2. Define a DuckDuckGo search function
def search_duckduckgo(query: str) -> str:
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=3)
        return "\n".join([f"{r['title']} - {r['href']}" for r in results])

# 3. Tools
tools = [
    Tool(
        name="DuckDuckGoSearch",
        func=search_duckduckgo,
        description="Useful for searching the web for current information."
    ),
    PythonREPLTool()
]

# 4. Agent
agent_executor = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# 5. Runner
def run_task(task: str):
    print(f"🧠 Executing Task: {task}\n")
    result = agent_executor.run(task)
    print(f"\n✅ Result:\n{result}")
    return result

# Example Usage
if __name__ == "__main__":
    run_task("Find the latest news on agentic AI and summarize it in 5 bullet points.")

