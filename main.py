import os
from typing import Any, Dict
from dotenv import load_dotenv
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import Tool, tool
from langchain.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults


load_dotenv()

@tool
def multiply(x: float, y: float) -> float:
    """Multiply 'x' times 'y'."""
    return x * y



if __name__ == "__main__":
    print("Tool Calling...")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "you are a helpful assistant"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ]
    )

    tools = [TavilySearchResults(), multiply]
    llm = ChatOpenAI(model="gpt-4o-mini")
    # llm = ChatAnthropic(model="claude-3-sonnet-20240229")

    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)

    res = agent_executor.invoke(
        {
            "input": "What is the weather in Dubai right now? Compare it with San Francisco, output should be in celsuis."
        }
    )
    print(res)