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

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                               model_kwargs={'device': "cpu"})
    vectorstore = FAISS.from_documents(documents=documents, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        temperature=0.4,
    )

    prompt = (
        """
        Sólo puedes responder en español. Utiliza el contexto proporcionado para responder a la pregunta. 
        Si la respuesta no se encuentra en el contexto, intenta responderla con tus propios conocimientos. 
        Si no sabes la respuesta, di que no lo sabes.
        \n\n
        {context}
        \n
        Pregunta: {input}
        \n
        Respuesta:
        """
    )

    prompt_template = PromptTemplate(
        input_variables=["context", "input"],
        template=prompt,
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return RagChainInterface(rag_chain)
