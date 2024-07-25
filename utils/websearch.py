from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate

from langchain.memory import ConversationSummaryBufferMemory
import os
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.retrievers.web_research import WebResearchRetriever
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, LLMChain, ConversationChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub, Cohere

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank

from langchain_community.retrievers import TavilySearchAPIRetriever

load_dotenv()

search = GoogleSearchAPIWrapper()
compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2")


def get_embedding():
    # embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        # model_kwargs=model_kwargs,
        # encode_kwargs={'normalize_embeddings': True}
    )
    return embeddings


def get_conversation_chain_v2(embeddings, llm=None):
    if llm is None:
        llm = Cohere(temperature=0.8, max_tokens=2048)

    # memory = ConversationBufferMemory(
    #     memory_key='chat_history', return_messages=True)

    prompt_template = """
    You are a helpful assistant named Linda. When user asks you a question, answer if you know the answer. 
    If you don't know the answer, use the following pieces of context from the web to answer the question at the end. 
    If you still don't know the answer, just say that you don't know, don't try to make up an answer. 

    {context} 

    Question: {question} 
    Helpful Answer:"""

    prompt_template_1 = """
    You are a helpful assistant named Linda. 
    Linda is talkative and provides lots of specific details from its context. 
    If you do not know the answer to a question, truthfully say you do not know. 
    Current conversation:
    {chat_history}
    
    Human: {input} 
    Linda:"""

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    prompt_1 = PromptTemplate(template=prompt_template_1, input_variables=['chat_history', 'input'])


    memory = ConversationSummaryBufferMemory(llm=llm, memory_key='chat_history', return_messages=True)

    # memory = ConversationBufferMemory(
    #     memory_key='chat_history', return_messages=True)

    vectorstore = Chroma(embedding_function=embeddings, persist_directory="./chroma_db_oai")

    web_research_retriever = WebResearchRetriever.from_llm(
    vectorstore=vectorstore,
    llm=llm,
    search=search,
    num_search_results=5,
    )

    # qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
    #     # combine_docs_chain_kwargs={"prompt": prompt},
    #     llm=llm,
    #     retriever=web_research_retriever, 
    #     memory=memory, 
    #     # return_source_documents=True, 
    #     max_tokens_limit=1024) 
    # # result = qa_chain({"question": user_input})

    compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=web_research_retriever
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        combine_docs_chain_kwargs={"prompt": prompt},
        llm=llm,
        retriever=compression_retriever,
        memory=memory,
        max_tokens_limit=1024,
    )

    non_retrieval_chain = ConversationChain(
        prompt=prompt_1, 
        llm=llm, 
        memory=memory,
        # max_tokens=2048,
        )


    return qa_chain, non_retrieval_chain


def get_conversation_chain_v1(embeddings, llm=None):
    if llm is None:
        llm = Cohere(temperature=0.8, max_tokens=2048)

    memory = ConversationBufferMemory(
        memory_key='chat_history', output_key='answer', return_messages=True)

    prompt_template = """
    You are a helpful assistant named Linda. When user asks you a question, use the following pieces of context to answer the question at the end. 
    If there are links in the context, keep the links as references.
    If you still don't know the answer, just say that you don't know, don't try to make up an answer. 

    {context} 

    Question: {question} 
    Helpful Answer with references:"""

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # vectorstore = Chroma(embedding_function=embeddings, persist_directory="./chroma_db_oai")

    # web_research_retriever = WebResearchRetriever.from_llm(
    # vectorstore=vectorstore,
    # llm=llm,
    # search=search,
    # num_search_results=5,
    # )

    web_research_retriever = TavilySearchAPIRetriever(k=5)

    compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=web_research_retriever
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        combine_docs_chain_kwargs={"prompt": prompt},
        llm=llm,
        retriever=compression_retriever,
        memory=memory,
        return_source_documents=True,
        # max_tokens_limit=1024,
    )

    return conversation_chain


