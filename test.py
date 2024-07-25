
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.retrievers.web_research import WebResearchRetriever
from langchain.vectorstores import Chroma

from langchain_cohere.embeddings import CohereEmbeddings

from langchain.chains import RetrievalQAWithSourcesChain, ConversationalRetrievalChain


from langchain.retrievers.web_research import WebResearchRetriever

from utils.utils import init_llm

from langchain_community.retrievers import TavilySearchAPIRetriever
from dotenv import load_dotenv

load_dotenv()

retriever = TavilySearchAPIRetriever(k=3)


# search = GoogleSearchAPIWrapper()

llm = init_llm()
# embeddings = CohereEmbeddings(model="embed-english-light-v3.0")

# vectorstore = Chroma(embedding_function=embeddings, persist_directory="./chroma_db_oai")

# web_research_retriever = WebResearchRetriever.from_llm(
# vectorstore=vectorstore,
# llm=llm,
# search=search,
# num_search_results=5,
# )

# user_input = "How do LLM Powered Autonomous Agents work?"
user_input = "what date is today?"
# qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm,retriever=retriever, return_source_documents=True) 

qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        # max_tokens_limit=1024,
    )


result = qa_chain.invoke({"question": user_input, 'chat_history': []})

# we get the results for user query with both answer and source url that were used to generate answer
# print("answer: +++++++++++++++++++", result["answer"])
# print("sources: -------------------", result["sources"])

print(result)


# docs = retriever.invoke(user_input)
# print(docs)
