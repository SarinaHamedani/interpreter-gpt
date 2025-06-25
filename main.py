import os
from dotenv import load_dotenv
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_experimental.tools import PythonREPLTool


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
        prompt=prompt,
        llm=ChatOpenAI(temperature=0, model="gpt-4o-mini"),
        tools=tools
    )
    python_agent_executor = AgentExecutor(agent=python_agent, tools=tools, verbose=True)
    python_agent_executor.invoke(
        input={
            "input": """generate and save in current working directory inside a folder called \"qrcodes\" 15 QRcodes 
            that point to https://github.com/SarinaHamedani/SarinaHamedani.github.io, you have qrcode package already installed."""
        }
    )

    csv_agent_executor: AgentExecutor = create_csv_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4o-mini"),
        path="episode_info.csv",
        verbose=True,
        allow_dangerous_code=True
    )
    csv_agent_executor.invoke(
        input={"input": "How many columns are there in file episode_info.csv"}
    )

    csv_agent_executor.invoke(
        {"input": "Print the seasons by ascending order of the number of episodes they have."}
    )


if __name__ == "__main__":
    main()
