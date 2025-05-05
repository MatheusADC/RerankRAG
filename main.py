from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate

from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

import os
os.environ['OPENAI_API_KEY'] = "your openai api key"

embeddings_model = OpenAIEmbeddings(model='text-embedding-3-small')
llm = ChatOpenAI(model_name='gpt-3.5-turbo', max_tokens=300)

pdf_link = 'your pdf link'
loader = PyPDFLoader(pdf_link, extract_images=False)
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=4000, 
    chunk_overlap=20,
    length_function=len,
    add_start_index=True
)

chunks = text_splitter.split_documents(pages)

vectordb = Chroma(embedding_function=embeddings_model, persist_directory='naive')

naive_retriever = vectordb.as_retriever(search_kwargs={"k": 10})

os.environ['COHERE_API_KEY'] = "your cohere api key"

rerank = CohereRerank(model="rerank-v3.5", top_n=3)

compressor_retriever = ContextualCompressionRetriever(
    base_compressor=rerank,
    base_retriever=naive_retriever,
)

TEMPLATE = """""
    Você é um especialista em legislação e tecnologia. Responda a pergunta abaiixo utilizando o contexto informado.
    Query:
    {question}

    Context:
    {context}
"""
rag_prompt = ChatPromptTemplate.from_template(TEMPLATE)

setup_retrival = RunnableParallel(
    {
        "question": RunnablePassthrough(),
        "context": compressor_retriever,
    }
)

output_parser = StrOutputParser()

compressor_retrieval_chain = setup_retrival | rag_prompt | llm | output_parser

compressor_retrieval_chain.invoke({"Quais são os principais pontos de risco do marco legal de IA"})
