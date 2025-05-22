from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph import MessagesState
from langgraph.types import Command
from pydantic import BaseModel, Field
from dotenv import load_dotenv, find_dotenv
from langgraph.prebuilt import create_react_agent
from langchain_tavily import TavilySearch
from langchain_gigachat.chat_models.gigachat import GigaChat
from tenacity import retry, stop_after_attempt, wait_fixed
from langchain.agents import tool

load_dotenv(find_dotenv())


def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()
            
class SearchInput(BaseModel):
    """Поисковый запрос"""
    query: str = Field(..., description="Текст поискового запроса")

search_tool = TavilySearch(
    max_results=5,
    topic="general",
    description="Используется для поиска данных в интернете",
    args_schema=SearchInput,
)

class Think(BaseModel):
    """Используется для рассуждений"""
    thought: str = Field(..., description="Твои рассуждения")

class Critic(BaseModel):
    """Используется для критики промежуточных результатов. Используй перечисления и для любых перечислений обязательно используй нумерацию."""
    critic: str = Field(..., description="Конструктивная и внимательная критика промежуточных результатов на следование КАЖДОМУ условию")

tools = [Think, Critic]

class CustomGigaChat(GigaChat):
    def invoke(self, *args, **kwargs):
        for attempt in range(5):
            try:
                result = super().invoke(*args, **kwargs)
                finish_reason = result.response_metadata['finish_reason']
                if finish_reason != 'length':
                    return result
                else:
                    print("!!! Length exceeded, retrying...")
                    continue
            except Exception as e:
                if attempt >= 4:
                    raise e
                else:
                    print(f"!!! Error: {e}, retrying...")
                    continue
        return result

giga = CustomGigaChat(model="GigaChat-2-Max",
                verify_ssl_certs=False,
                profanity_check=False,
                top_p=0,
                base_url="https://gigachat.ift.sberdevices.ru/v1",
                streaming=False,
                max_tokens=2000,
                # repetition_penalty=1,
                timeout=30)



config={"recursion_limit": 1000}

system = """You have access to a "think" tool that provides a dedicated space for structured reasoning. Using this tool significantly improves your performance on complex tasks. 

## When to use the think tool 
Before taking any action or responding to the user after receiving tool results, use the think tool as a scratchpad to: 
- List the specific rules that apply to the current request 
- Check if all required information is collected 
- Verify that the planned action complies with all policies 
- Iterate over tool results for correctness 
- Analyze complex information from web searches or other tools 
- Plan multi-step approaches before executing them 

## How to use the think tool effectively 
When using the think tool: 
1. Break down complex problems into clearly defined steps 
2. Identify key facts, constraints, and requirements 
3. Check for gaps in information and plan how to fill them 
4. Evaluate multiple approaches before choosing one 
5. Verify your reasoning for logical errors or biases"""

tools = [Think, Critic]

# agent = create_react_agent(giga, tools=tools, prompt=system)

def create_agent():
    return create_react_agent(giga, tools=tools, prompt=system)

# inputs = {"messages": [("user", "У Алисы было 2 брата и одна сестра. Сколько сестер у братьев алисы?")]}
# print_stream(agent.stream(inputs, config=config, stream_mode="values"))
