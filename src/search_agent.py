from dotenv import find_dotenv, load_dotenv
from langchain_gigachat.chat_models.gigachat import GigaChat
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

load_dotenv(find_dotenv())


# def print_stream(stream):
#     for s in stream:
#         message = s["messages"][-1]
#         if isinstance(message, tuple):
#             print(message)
#         else:
#             message.pretty_print()

class SearchInput(BaseModel):
    """Поисковый запрос"""
    query: str = Field(..., description="Текст поискового запроса")

search_tool = TavilySearch(
    max_results=5,
    topic="general",
    description="Используется для поиска данных в интернете",
    args_schema=SearchInput,
)

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



config={"recursion_limit": 10}

system = """Ты полезный ассистент"""

tools = [search_tool]

agent = create_react_agent(giga, tools=tools, prompt=system)

# def create_agent():
#     return create_react_agent(giga, tools=tools, prompt=system)

# inputs = {"messages": [("user", "Какая погода завтра в Москве?")]}
# print_stream(agent.stream(inputs, config={}, stream_mode="values"))
