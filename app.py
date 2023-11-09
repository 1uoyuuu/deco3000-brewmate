# Standard Python libraries for system and environment interaction
import os  # Interface with the operating system
import re  # Regular expressions library

# JSON and HTTP handling
import json  # JSON encoder and decoder
import requests  # HTTP library for making requests

# Parsing HTML data
from bs4 import BeautifulSoup  # Library for pulling data out of HTML and XML files

# Streamlit libraries for creating web UI components
import streamlit as st  # Main Streamlit library for web apps
from streamlit_chat import message  # Streamlit chat widget for creating chat interfaces
from streamlit_echarts import (
    st_echarts,
)  # Streamlit integration for creating charts using ECharts

# Environment configuration
from dotenv import load_dotenv  # Loads environment variables from a .env file

# Typing for better type hints in functions
from typing import (
    List,
    Union,
    Type,
)  # Typing for function signatures and complex type hints

# Pydantic for data parsing and validation using Python type annotations
from pydantic import (
    BaseModel,
    Field,
)  # Base model for data classes and field definition

# LangChain libraries for AI and NLP processing
from langchain.document_loaders.csv_loader import (
    CSVLoader,
)  # Loader for handling CSV files
from langchain.vectorstores.faiss import FAISS  # Vector storage and search with FAISS
from langchain.embeddings.openai import (
    OpenAIEmbeddings,
)  # Embeddings from OpenAI models

# Utilities for search and prompt templating
from langchain.utilities import (
    GoogleSerperAPIWrapper,
)  # Wrapper for Google Search Engine Results Page API
from langchain.prompts import (
    PromptTemplate,
    StringPromptTemplate,
)  # Templates for creating prompts for language models

# Base tool for extending LangChain capabilities
from langchain.tools import BaseTool  # Base class for creating new tools

# Schema definitions and text splitting utilities
from langchain.schema import (
    AgentAction,
    AgentFinish,
    OutputParserException,
    SystemMessage,
)  # Schema definitions for agent actions
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
)  # Utility for splitting large texts into manageable parts

# AI agent initializers, types, and execution tools
from langchain.agents import (
    initialize_agent,  # Function to initialize an agent
    Tool,  # Base class for creating tools to work with agents
    AgentType,  # Enum type to define agent types
    AgentExecutor,  # Executor for running agent actions
    LLMSingleActionAgent,  # Single action agent from LangChain
    AgentOutputParser,  # Parser for handling agent output
)

# LangChain chains and memory management for conversational contexts
from langchain.chains import LLMChain  # Chain for linearly running LLM tasks
from langchain.chains.summarize import (
    load_summarize_chain,
)  # Specific chain for summarization tasks
from langchain.chains.openai_functions.extraction import (
    create_extraction_chain,
)  # Chain for extraction tasks using OpenAI functions
from langchain.memory import (
    ConversationBufferWindowMemory,
)  # Memory buffer for managing conversation context
from langchain.chat_models import (
    ChatOpenAI,
)  # OpenAI's chat model integrated within LangChain


# Environment Configuration
load_dotenv()  # Load environment variables from a .env file
browserless_api_key = os.getenv(
    "BROWSERLESS_API_KEY"
)  # Retrieve the API key for Browserless service
serper_api_key = os.getenv("SERPER_API_KEY")  # Retrieve the API key for Serper service


# Similarity Search Function
def retrieve_coffee(input):
    # Load documents from a CSV file for the database
    loader = CSVLoader(file_path="coffee-database.csv")
    documents = loader.load()

    # Create embeddings for the documents using OpenAI models
    embeddings = OpenAIEmbeddings()

    # Use FAISS for efficient similarity search in the embedding space
    db = FAISS.from_documents(documents, embeddings)

    # Retrieve the most similar document based on the input
    similar_response = db.similarity_search(
        input, k=1
    )  # For simplicity we only retreive the most similar result to accelerate the process

    # Extract the page content from the similar document
    page_content_array = [doc.page_content for doc in similar_response]

    # Find URLs using regex in the page content
    url_pattern = re.compile(r"https?://[^\s]+")
    urls = [url_pattern.findall(s) for s in page_content_array]

    # Flatten the list of URLs since we expect one URL per document
    urls = [url for sublist in urls for url in sublist]

    return urls if urls else None


# This function is used to extract coffee information with a JSON format,
# It is useful to process the unstructured text content scraped from the website
# The schema is set to focus on extracting coffee information only
def coffee_info_extract(content: str):
    schema = {
        "properties": {
            "coffee_name": {
                "type": "string",
                "description": "The official name of the coffee product.",
            },
            "roaster": {
                "type": "string",
                "description": "The company or individual who roasted the coffee beans, often corresponding to the website where the coffee can be purchased.",
            },
            "green_bean_producer": {
                "type": "string",
                "description": "The farmer or company responsible for cultivating and harvesting the green coffee beans.",
            },
            "origin": {
                "type": "object",
                "description": "The geographical location where the coffee is sourced, including mandatory country and optional region and farm.",
                "properties": {
                    "country": {
                        "type": "string",
                        "description": "The country where the coffee is sourced.",
                    },
                    "region": {
                        "type": "string",
                        "description": "The specific region within the country where the coffee is sourced.",
                    },
                    "farm": {
                        "type": "string",
                        "description": "The name of the farm where the coffee is grown.",
                    },
                },
                "required": ["country", "region", "farm"],
            },
            "variety": {
                "type": "string",
                "description": "The specific variety or cultivar of the coffee plant.",
            },
            "processing": {
                "type": "string",
                "description": "Details about the processing method used for the coffee after harvest.",
            },
            "flavor_profile": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of primary flavor notes identified in the coffee.",
            },
            "community_support": {
                "type": "string",
                "description": "Information about the producer's involvement and support for local communities.",
            },
            "brewing_guidelines": {
                "type": "object",
                "properties": {
                    "filter": {
                        "type": "object",
                        "properties": {
                            "ratio": {
                                "type": "string",
                                "description": "The coffee-to-water ratio recommended for filter brewing.",
                            },
                            "grind_size": {
                                "type": "string",
                                "description": "The recommended grind size for filter brewing.",
                            },
                            "temperature": {
                                "type": "string",
                                "description": "The optimal water temperature for filter brewing.",
                            },
                            "time": {
                                "type": "string",
                                "description": "The recommended brew time for filter brewing.",
                            },
                        },
                        "required": ["ratio", "grind_size", "temperature", "time"],
                    },
                    "espresso": {
                        "type": "object",
                        "properties": {
                            "dose": {
                                "type": "string",
                                "description": "The amount of coffee used for a single espresso shot.",
                            },
                            "yield": {
                                "type": "string",
                                "description": "The expected volume of espresso extracted.",
                            },
                            "time": {
                                "type": "string",
                                "description": "The extraction time for the espresso shot.",
                            },
                        },
                        "required": ["dose", "yield", "time"],
                    },
                },
                "required": ["filter", "espresso"],
                "description": "Recommended brewing guidelines for both filter and espresso methods.",
            },
            "cup_scores": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "score": {
                            "type": "number",
                            "description": "The score given to the coffee during cupping sessions.",
                        },
                        "event": {
                            "type": "string",
                            "description": "The event or competition where the coffee was scored.",
                        },
                    },
                    "required": ["score", "event"],
                },
                "description": "Scores received from various cupping competitions.",
            },
            "url": {
                "type": "string",
                "description": "The URL where the coffee can be purchased or more information can be found.",
            },
        },
        "required": [
            "coffee_name",
            "roaster",
            "producer",
            "origin",
            "variety",
            "processing",
            "flavor_profile",
            "brewing_guidelines",
            "url",
        ],
    }
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
    return create_extraction_chain(schema=schema, llm=llm).run(content)


# Scraping Website Function
def scrape_website(url: str):
    headers = {
        "Cache-Control": "no-cache",
        "Content-Type": "application/json",
    }
    data = {"url": url}
    data_json = json.dumps(data)
    post_url = f"https://chrome.browserless.io/content?token={browserless_api_key}"
    response = requests.post(post_url, headers=headers, data=data_json)

    if response.status_code == 200:
        print("Scraping website succeed...")
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()
        text_without_whitespaces = re.sub(r"\s+", " ", text).strip()
        final_text = f"URL: {url} {text_without_whitespaces}"
        print(len(final_text))
        if len(final_text) > 10000:
            return final_text[:10000]  # Join the first 10000 character
        else:
            return final_text  # If less than 10000 words, return the original text
    else:
        # Enhanced error logging
        print(f"HTTP request failed with status code {response.status_code}")
        print(f"Response body: {response.text}")  # Log the full response body
        # Optionally, you could raise an exception here or handle the error as appropriate


def coffee_recommendation_agent(user_input: str):
    # Tool for coffee recommendation agent, it needs to retrieve coffee from the database,
    # analyse the url with scraper tool, and extract relevant information
    tools = [
        Tool(
            name="Retrieval",
            func=retrieve_coffee,
            description="useful for you need to retireve coffee options based on user flavour profiles, IT CAN BE CALLED ONLY ONCE and it should be the FIRST TOOL to use. It will retrieve URLs of the top three most similar coffee entries from the database. Input should be A string ONLY representing the user's flavor preference",
        ),
        Tool(
            name="Scraper",
            func=scrape_website,
            description="useful for when you are provided a url and you want to look through the detail of a specific site. Must be called after Coffee Retrieval is called. Input should be only the url, make sure to remove additional quotation mark",
        ),
    ]
    system_message = SystemMessage(
        content="""You are a professional barista, who can do tailored recommendation on specialty coffee that match users preference. 
    You do not make things up, you will try as hard as possible to gather actual coffee data.

    Please make sure you complete the recommendation above with the following rules:
    1/ You should start with retrieving relevant coffee from the database with the flavour profiles users would like
    2/ If there are url of relevant links, you will scrape it to gather more information
    3/ You should scrape all the relevant links you have to gain more comprehensive undertanding, But don't do this more than 3 iteratins
    4/ In the final output, present a detailed explanation for your coffee recommendation and make sure to provide everything you know about the coffee, you should include the link to back up your recommendation.
    """
    )

    agent_kwargs = {
        "system_message": system_message,
    }
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        agent_kwargs=agent_kwargs,
    )
    result = agent({"input": user_input})
    # the output of this agent is a dict, we only want to keep the output str
    return result["output"]


def research_agent(user_input: str):
    def search(query):
        url = "https://google.serper.dev/search"

        payload = json.dumps({"q": query})

        headers = {"X-API-KEY": serper_api_key, "Content-Type": "application/json"}

        response = requests.request("POST", url, headers=headers, data=payload)

        return response.text

    # 2. Tool for scraping
    def scrape_website(objective: str, url: str):
        # scrape website, and also will summarize the content based on objective if the content is too large
        # objective is the original objective & task that user give to the agent, url is the url of the website to be scraped

        print("Scraping website...")
        # Define the headers for the request
        headers = {
            "Cache-Control": "no-cache",
            "Content-Type": "application/json",
        }

        # Define the data to be sent in the request
        data = {"url": url}

        # Convert Python object to JSON string
        data_json = json.dumps(data)

        # Send the POST request
        post_url = f"https://chrome.browserless.io/content?token={browserless_api_key}"
        response = requests.post(post_url, headers=headers, data=data_json)

        # Check the response status code
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            text = soup.get_text()
            if len(text) > 10000:
                output = summary(objective, text)
                return output
            else:
                return text
        else:
            print(f"HTTP request failed with status code {response.status_code}")

    def summary(objective, content):
        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500
        )
        docs = text_splitter.create_documents([content])
        map_prompt = """
        Write a summary of the following text for {objective}:
        "{text}"
        SUMMARY:
        """
        map_prompt_template = PromptTemplate(
            template=map_prompt, input_variables=["text", "objective"]
        )

        summary_chain = load_summarize_chain(
            llm=llm,
            chain_type="map_reduce",
            map_prompt=map_prompt_template,
            combine_prompt=map_prompt_template,
            verbose=True,
        )

        output = summary_chain.run(input_documents=docs, objective=objective)

        return output

    class ScrapeWebsiteInput(BaseModel):
        """Inputs for scrape_website"""

        objective: str = Field(
            description="The objective & task that users give to the agent"
        )
        url: str = Field(description="The url of the website to be scraped")

    class ScrapeWebsiteTool(BaseTool):
        name = "scrape_website"
        description = "useful when you need to get data from a website url, passing both url and objective to the function; DO NOT make up any url, the url should only be from the search results"
        args_schema: Type[BaseModel] = ScrapeWebsiteInput

        def _run(self, objective: str, url: str):
            return scrape_website(objective, url)

        def _arun(self, url: str):
            raise NotImplementedError("error here")

    # 3. Create langchain agent with the tools above
    tools = [
        Tool(
            name="Search",
            func=search,
            description="useful for when you need to answer questions about current events, data. You should ask targeted questions",
        ),
        ScrapeWebsiteTool(),
    ]

    system_message = SystemMessage(
        content="""You are a world class researcher, who can do detailed research on any topic and produce facts based results; 
                you do not make things up, you will try as hard as possible to gather facts & data to back up the research
                
                Please make sure you complete the objective above with the following rules:
                1/ You should do enough research to gather as much information as possible about the objective
                2/ If there are url of relevant links & articles, you will scrape it to gather more information
                3/ After scraping & search, you should think "is there any new things i should search & scraping based on the data I collected to increase research quality?" If answer is yes, continue; But don't do this more than 2 iteratins
                4/ You should not make things up, you should only write facts & data that you have gathered
                5/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research
                6/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research"""
    )

    agent_kwargs = {
        "system_message": system_message,
    }

    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        agent_kwargs=agent_kwargs,
    )
    result = agent({"input": user_input})
    # the output of this agent is a dict, we only want to keep the output str
    return result["output"]


# Combination of scraping and extracting
def scrape_and_extract(url: str):
    url = url.strip('"').strip("'")
    content = scrape_website(url)
    result = coffee_info_extract(content)
    return result


# Tool declaration for main agent
tools = [
    Tool(
        name="Coffee Recommendation",
        func=coffee_recommendation_agent,
        description="useful for recommending coffee based on a user's flavour preferences. Don't use this tool until the user is specifically asking for coffee recommendation. The input should be the flavour profiles only.",
        return_direct=True,
    ),
    Tool(
        name="Coffee Info Extraction",
        func=scrape_and_extract,
        description="useful for when you are provided an actual link of coffee product and you want to extract the coffee information from the website (ONLY USE THIS ONE WHEN USER PROVIDES YOU WITH URL, DON'T MAKE UP YOUR OWN URL). INPUT SHOULD BE the url, removing additional quotation mark",
    ),
    Tool(
        name="Coffee Researcher",
        func=research_agent,
        description="useful for providing educational content based on the questoin being asked, \
                    it will try as hard as possible to conduct a research on given topic. \
                    USE IT ONLY AFTER YOU HAVE TRIED Google Search ONCE",
        return_direct=True,
    ),
    Tool(
        name="Google Search",
        func=GoogleSerperAPIWrapper().run,
        description="useful for obtaining general information about current events or answer simple questions. ALWAYS USE THIS AS THE FIRST TOOL TO ANSWER QUESTIONS",
    ),
]


# Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in self.tools]
        )
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)


class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise OutputParserException(
                f"Could not parse FLAVOURHUNT LLM output: `{llm_output}`"
            )
        action = match.group(1).strip()
        action_input = match.group(2)
        return AgentAction(tool=action, tool_input=action_input, log=llm_output)


output_parser = CustomOutputParser()


# Set up the base template
template_with_history = """Answer the following questions as best you can, speaking as passionate coffee professional. You have access to the following tools:

{tools}

Here are some basic rule you need to follow:
NEVER USE Coffee Recommendation IF USER DOESN'T EXPLICITY ASK FOR A COFFEE RECOMMENDATION.
ALWAYS START WITH Google Search to provide basic answer to questions, and ONLY USE Coffee Researcher when you can't find good search result or user want to know dive deeper into certain topic. 
ONLY USE Coffee Info Extraction when user provides an actual link or url.


And You must use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]. 
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin! Remember to answer as a trustworthy and knowledgeable coffee professional when giving your final answer,
and always try to explain your answer since the user is new to the coffee world.

Previous conversation history:
{history}

New question: {input}
{agent_scratchpad}"""


prompt_with_history = CustomPromptTemplate(
    template=template_with_history,
    tools=tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps", "history"],
)


# Define a function to display the CoffeeGPT page
def coffee_gpt_page():
    st.title("☕CoffeeGPT")
    st.markdown(
        """
            > :black[**Transforming specialty coffee experience, one cup at a time.**]
            """
    )
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    llm_chain = LLMChain(llm=llm, prompt=prompt_with_history)

    tool_names = [tool.name for tool in tools]
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=["\nObservation:"],
        allowed_tools=tool_names,
    )

    memory = ConversationBufferWindowMemory(k=2)
    if "memory" not in st.session_state:
        st.session_state.memory = memory

    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        memory=st.session_state.memory,
        handle_parsing_errors="Check your output and make sure it conforms correctly!",
    )
    # Get the user input
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key="my_form", clear_on_submit=True):
            user_input = st.text_input(
                "Question:",
                placeholder="Ask anything about coffee",
                key="input",
            )
            submit_button = st.form_submit_button(label="Send")

        if submit_button and user_input:
            output = agent_executor.run(user_input)
            st.session_state["past"].append(user_input)
            st.session_state["generated"].append(output)

    if st.session_state["generated"]:
        with reply_container:
            for i in range(len(st.session_state["generated"])):
                message(
                    st.session_state["past"][i],
                    is_user=True,
                    key=str(i) + "_user",
                    avatar_style="pixel-art",
                )
                message(
                    st.session_state["generated"][i],
                    key=str(i),
                    avatar_style="bottts",
                )


# Define a function to display the Flavour Wheel page
def flavour_wheel_page():
    st.header("Coffee Flavour Wheel")
    with open("drink-flavors.json", "r") as f:
        data = json.loads(f.read())

    option = {
        "title": {
            "text": "WORLD COFFEE RESEARCH SENSORY LEXICON",
            "subtext": "Source: https://worldcoffeeresearch.org/work/sensory-lexicon/",
            "textStyle": {"fontSize": 14, "align": "center"},
            "subtextStyle": {"align": "center"},
            "sublink": "https://worldcoffeeresearch.org/work/sensory-lexicon/",
        },
        "series": {
            "type": "sunburst",
            "data": data,
            "radius": [0, "95%"],
            "sort": None,
            "emphasis": {"focus": "ancestor"},
            "levels": [
                {},
                {
                    "r0": "15%",
                    "r": "35%",
                    "itemStyle": {"borderWidth": 2},
                    "label": {"rotate": "tangential"},
                },
                {"r0": "35%", "r": "70%", "label": {"align": "right"}},
                {
                    "r0": "70%",
                    "r": "72%",
                    "label": {"position": "outside", "padding": 3, "silent": False},
                    "itemStyle": {"borderWidth": 3},
                },
            ],
        },
    }
    st_echarts(option, height="700px")


def main():
    st.sidebar.title("Navigation")
    # Define the navigation structure
    pages = {
        "CoffeeGPT": coffee_gpt_page,
        "Coffee Flavour Wheel": flavour_wheel_page,
    }

    # Radio buttons for navigation
    page = st.sidebar.radio("Select a page:", tuple(pages.keys()))

    # Display the selected page with the radio buttons
    pages[page]()


if __name__ == "__main__":
    st.set_page_config(page_title="CoffeeGPT")

    if "generated" not in st.session_state:
        st.session_state["generated"] = [
            "Hi! I'm CoffeeGPT, an intelligent chatbot that designed to elevate your coffee experience. ☕"
        ]
    if "past" not in st.session_state:
        st.session_state["past"] = ["👋 Hello!"]

    main()
