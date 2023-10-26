import streamlit as st

from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv

# load the environment configuration
load_dotenv()

# 1. vectorise all the coffee csv data
loader = CSVLoader(file_path="data-v2.csv")
documents = loader.load()
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)


# 2. similarity search function
def retrieve_info(query):
    similar_response = db.similarity_search(
        query, k=3
    )  # return top 3 similar results but i think we only need one
    page_content_array = [doc.page_content for doc in similar_response]
    return page_content_array


# # ------------------DISCARD------------------
# message = (
#     "I want some thing tastes like tamarind juice, cherry, passionfruit, chocolate"
# )

# response = retrieve_info(message)

# for item in response:
#     print(item)
# # ------------------DISCARD------------------

# 3. setup llm & prompt engineering
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

template = """
You are a sophisticated coffee recommendation engine.
A user will share their preferred coffee flavours with you, and your task is to provide the most suitable coffee recommendation based on the existing coffee options in our database.
Please adhere to ALL of the following guidelines:

1/ Your recommendation should closely or identically match a coffee option that exists in our database in terms of tasting notes, aroma, and other flavour profiles.

2/ If there is no identical match, try to find a coffee that has the most similar tasting notes that align closely with the user's stated preferences.

Below is the information I have received from the user about their flavour preferences:
{message}

Here is the list of coffee options available in our database:
{recommendations}

Please write the best coffee recommendation that I should send to this user:
"""

prompt = PromptTemplate(
    input_variables=["message", "recommendations"], template=template
)

chain = LLMChain(llm=llm, prompt=prompt)


# 4. retrieval augmented generation
def generate_recommendation(message):
    recommendations = retrieve_info(message)
    response = chain.run(message=message, recommendations=recommendations)
    return response


# 5. Build an app with streamlit
def main():
    st.set_page_config(
        page_title="Coffee Recommendation Generator", page_icon=":coffee:"
    )

    st.header("Coffee Recommendation Generator :coffee:")
    message = st.text_area("Flavour Preferences")

    if st.button("Generate", type="primary"):
        st.write("Generating the best possible coffee that suits your tastebuds...")
        result = generate_recommendation(message)
        st.info(result)


if __name__ == "__main__":
    main()
