import llama_index.core.llms.llm
from dotenv import load_dotenv
import os
import pandas as pd
from llama_index.experimental.query_engine import PandasQueryEngine
from prompts import new_prompt, instruction_str, context
from note_engine import note_engine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.legacy.agent.openai.base import OpenAI
import openai


load_dotenv()

# Specify path to data
population_path = os.path.join("data", "population.csv")
# load data in pandas
population_df = pd.read_csv(population_path)

population_query_engine = PandasQueryEngine(df=population_df, verbose=True, instruction_str=instruction_str)


population_query_engine.update_prompts({"pandas_prompt": new_prompt})
#population_query_engine.query("what is the population of Denmark")

# specify tools that we have access to
tools = [
    note_engine,
    QueryEngineTool(query_engine=population_query_engine,
    metadata=ToolMetadata(name="population-data",description="gives information of the world population and demographics",),
     ),

]
llm = OpenAI(model="gpt-4")
agent = ReActAgent.from_tools(tools, llm=llm, verbose=True, context=context)

while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    result = agent.query(prompt)
    print(result)

