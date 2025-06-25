import os
from typing import Any, Dict
from dotenv import load_dotenv
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_experimental.agents.agent_toolkits.pandas.base import (
    create_pandas_dataframe_agent,
)
from langchain_experimental.tools import PythonREPLTool
from langchain_core.tools import Tool
import pandas as pd


load_dotenv()


def main():
    print("Start...")

    instructions = """
    You are an agent designed to write and execute python code to answer questions.
    You have access to a python REPL, which you can use to execute python code.
    If you get an error, debug your code and try again.
    Only use the output of your code to answer the question. 
    You might know the answer without running any code, but you should still run the code to get the answer.
    If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
    """
    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instructions=instructions)

    tools = [PythonREPLTool()]
    python_agent = create_react_agent(
        prompt=prompt, llm=ChatOpenAI(temperature=0, model="gpt-4o-mini"), tools=tools
    )
    python_agent_executor = AgentExecutor(agent=python_agent, tools=tools, verbose=True)

    def python_agent_executor_wrapper(original_prompt: str) -> Dict[str, Any]:
        return python_agent_executor.invoke({"input": original_prompt})

    df = pd.read_csv("episode_info.csv")

    csv_agent_executor: AgentExecutor = create_pandas_dataframe_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4o-mini"),
        df=df,
        verbose=True,
        allow_dangerous_code=True,
    )

    ################################### Router Grand Agent ###################################
    tools = [
        Tool(
            name="Python Executor Agent",
            func=python_agent_executor_wrapper,
            description="""Useful when you need to transform natural language to python and execute the python code, 
            returning the resuls of the code execution
            DOES NOT ACCEPT CODE AS INPUT
            """,
        ),
        Tool(
            name="CSV Agent",
            func=csv_agent_executor.invoke,
            description="""Useful when you need tyo answer question over episode_info.csv file,
            takes an input the entire question and returns the answer after running pandas calculations
            """,
        ),
    ]

    prompt = base_prompt.partial(instructions="")
    grand_agent = create_react_agent(
        prompt=prompt, llm=ChatOpenAI(temperature=0, model="gpt-4o-mini"), tools=tools
    )

    grand_agent_executor = AgentExecutor(agent=grand_agent, tools=tools, verbose=True)

    print(grand_agent_executor.invoke({"input": "Which season has the most episodes"}))

    print(
        grand_agent_executor.invoke(
            {
                "input": 'generate and save in current working directory inside a folder called "qrcodes" 15 QRcodes that point to https://github.com/SarinaHamedani/SarinaHamedani.github.io, you have qrcode package already installed.'
            }
        )
    )


if __name__ == "__main__":
    main()
