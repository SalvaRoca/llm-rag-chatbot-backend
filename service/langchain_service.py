import os, dotenv

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import TokenTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from .rag_chain_interface import RagChainInterface


def load_rag_chain(llm_ref):
    dotenv.load_dotenv()

    documents = []

    text_splitter = TokenTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    for filename in os.listdir('data'):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join('data', filename)
            loader = PyPDFLoader(pdf_path)
            chunks = loader.load_and_split(text_splitter=text_splitter)
            documents.extend(chunks)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents=documents, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    if llm_ref == 'mistral':
        repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
        prompt = (
            """
            [INST]Utilizando la siguiente información de contexto: {context}
            Y teniendo en cuenta el historial de la conversación en el que yo soy el Usuario: {history}
            Responde en español a la siguiente petición. Si la pregunta no está relacionada con el contexto o la 
            respuesta a la misma no se encuentra en el contexto, respóndela con tus propios conocimientos ignorando el 
            contexto: {input} 
            Responde a partir de aquí en español y usando notación Markdown:[/INST]
            """
        )
    else:
        repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        prompt = (
            """
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            Utilizando la siguiente información de contexto: {context}
            Y teniendo en cuenta el historial de la conversación en el que yo soy el Usuario: {history}
            Responde en español a la siguiente petición. Si la pregunta no está relacionada con el contexto o la 
            respuesta a la misma no se encuentra en el contexto, respóndela con tus propios conocimientos ignorando el 
            contexto:
            <|start_header_id|>user<|end_header_id|>
            {input} 
            Responde a partir de aquí en español y usando notación Markdown:
            <|start_header_id|>assistant<|end_header_id|>
            """
        )

    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        max_length=512,
        temperature=0.35,
        top_p=0.95,
        repetition_penalty=1.2,
        cache=False,
        model_kwargs={
            "length_penalty": 1.3
        }
    )

    prompt_template = PromptTemplate(
        input_variables=["context", "input", "history"],
        template=prompt,
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return RagChainInterface(rag_chain)
