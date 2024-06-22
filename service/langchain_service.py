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


def load_rag_chain(repo_id):
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

    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        temperature=0.4,
        top_p=0.95,
        repetition_penalty=1.05,
        cache=False,
        model_kwargs={
            "length_penalty": 1.3
        }
    )

    prompt = (
        """
        Utiliza el contexto proporcionado para responder a la pregunta. 
        Si la respuesta no se encuentra en el contexto, intenta responderla con tus propios conocimientos descartando la
        información del contexto. Si no sabes la respuesta, di que no lo sabes. 
        \n
        Contexto:
        {context}
        \n
        Historial de la conversación (tú eres el Bot):
        {history}
        \n
        Debes responder sólo a la siguiente pregunta en español: {input}
        \n
        Escribe tu respuesta a continuación:
        """
    )

    prompt_template = PromptTemplate(
        input_variables=["context", "input", "history"],
        template=prompt,
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return RagChainInterface(rag_chain)

