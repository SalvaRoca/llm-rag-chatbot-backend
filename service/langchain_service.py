import os, dotenv
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def load_rag_chain(repo_id):
    dotenv.load_dotenv()

    # Definir el objeto TextSplitter
    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )

    # Lista para almacenar los chunks de texto de todos los archivos PDF
    all_chunks = []

    # Iterar a través de cada archivo en la carpeta 'data'
    for filename in os.listdir('data'):
        # Verificar si el archivo es un PDF
        if filename.endswith('.pdf'):
            # Construir la ruta relativa del archivo PDF
            pdf_path = os.path.join('data', filename)

            # Cargar el archivo PDF
            loader = PyPDFLoader(pdf_path)

            # Dividir el archivo PDF en chunks de texto
            chunks = loader.load_and_split(text_splitter=text_splitter)

            # Agregar los chunks de texto a la lista
            all_chunks.extend(chunks)

    embeddings = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                               model_kwargs={'device': "cpu"})

    vectorstore = FAISS.from_documents(documents=all_chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    llm = HuggingFaceEndpoint(repo_id=repo_id, temperature=0.3)

    prompt = hub.pull("rlm/rag-prompt")

    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    return rag_chain
