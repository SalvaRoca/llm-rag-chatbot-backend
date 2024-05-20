import os
from llama_index.core import SimpleDirectoryReader, Document, PromptHelper, ServiceContext
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.huggingface import HuggingFaceInferenceAPI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import StorageContext
import faiss
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.core.callbacks import CallbackManager


def format_docs():
    documents = SimpleDirectoryReader(input_files='/data'.load_data())
    doc_text = "\n\n".join([d.get_content() for d in documents])
    return [Document(text=doc_text)]


def load_docs_from_folder():
    all_chunks = []
    for filename in os.listdir('./data'):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join('./data', filename)
            documents = SimpleDirectoryReader(input_files=[pdf_path]).load_data()
            doc_text = "\n\n".join([d.get_content() for d in documents])
            text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
            chunks = text_splitter.split_text(doc_text)
            for chunk in chunks:
                all_chunks.append(Document(text=chunk))
    return all_chunks



def load_rag_chain(repo_id):
    node_parser = SimpleNodeParser.from_defaults()
    all_chunks = load_docs_from_folder()

    base_nodes = node_parser.get_nodes_from_documents(all_chunks)

    for idx, node in enumerate(base_nodes):
        node.id_ = f"node-{idx}"

    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    llm = HuggingFaceInferenceAPI(model_name=repo_id, embedding_dim=1536)
    service_context = ServiceContext.from_defaults(llm=llm, embed_model=Settings.embed_model)

    d = 384
    faiss_index = faiss.IndexFlatL2(d)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    callback_manager = CallbackManager()

    index = VectorStoreIndex.from_documents(all_chunks, service_context=service_context)
    query_engine = index.as_query_engine(similarity_top_k=2)
    return query_engine