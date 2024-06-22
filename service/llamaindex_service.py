import os, dotenv

from llama_index.core import (
    SimpleDirectoryReader,
    Document,
    VectorStoreIndex,
    ServiceContext,
    StorageContext,
    ChatPromptTemplate
)
from llama_index.llms.huggingface import HuggingFaceInferenceAPI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.text_splitter import TokenTextSplitter
import faiss
from .rag_chain_interface import RagChainInterface


def load_rag_chain(repo_id):
    dotenv.load_dotenv()

    documents = []

    text_splitter = TokenTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    for filename in os.listdir('./data'):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join('./data', filename)
            documents = SimpleDirectoryReader(input_files=[pdf_path]).load_data()
            doc_text = "\n\n".join([d.get_content() for d in documents])
            text_splitter = text_splitter
            chunks = text_splitter.split_text(doc_text)
            for chunk in chunks:
                documents.append(Document(text=chunk))

    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    llm = HuggingFaceInferenceAPI(
        model_name=repo_id,
        generate_kwargs={
            "temperature": 0.4,
            "max_length": 512,
            "top_p": 0.95,
            "repetition_penalty": 1.05,
            "length_penalty": 1.3,
            "use_cache": False
        }
    )

    service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)

    vector_store = FaissVectorStore(faiss_index=faiss.IndexFlatL2(384))
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, service_context=service_context)

    prompt = (
        """
        Contexto:
        {context_str}
        \n
        Debes responder sólo a la siguiente pregunta en español: {query_str}
        \n
        Escribe tu respuesta a continuación:
        """
    )

    chat_text_qa_msgs = [
        (
            "system",
            """ 
            Sólo puedes responder en español. Utiliza el contexto proporcionado para responder a la pregunta.
            Si la respuesta no se encuentra en el contexto, intenta responderla con tus propios conocimientos
            descartando la información del contexto. Si no sabes la respuesta, di que no lo sabes.
            """
        ),
        ("user", prompt),
    ]

    text_qa_template = ChatPromptTemplate.from_messages(chat_text_qa_msgs)

    rag_chain = index.as_query_engine(similarity_top_k=2, text_qa_template=text_qa_template)

    return RagChainInterface(rag_chain)
