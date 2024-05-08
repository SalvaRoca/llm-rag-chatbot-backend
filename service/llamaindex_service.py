from llama_index.core import SimpleDirectoryReader, PromptHelper
from llama_index.core import Document
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.huggingface import HuggingFaceInferenceAPI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import StorageContext
import faiss
import os
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.core.callbacks import CallbackManager

# Cargar documentos
documents = SimpleDirectoryReader(input_files=["tema3.pdf"]).load_data()

# Combinar documentos en uno solo
doc_text = "\n\n".join([d.get_content() for d in documents])
text = [Document(text=doc_text)]

# Configuración del parser de nodos
node_parser = SimpleNodeParser.from_defaults() # Default chunk size is 1024

# Crear nodos a partir del texto
base_nodes = node_parser.get_nodes_from_documents(text)

# Reiniciar IDs de nodos
for idx, node in enumerate(base_nodes):
    node.id_ = f"node-{idx}"

# Configuración del modelo de lenguaje
hf_token = "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"
os.environ["HF_TOKEN"] = hf_token
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Configuración del contexto del servicio
llm = HuggingFaceInferenceAPI(model_name="mistralai/Mistral-7B-Instruct-v0.2", token=hf_token,    embedding_dim=1536
)
text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
prompt_helper = PromptHelper(
    context_window=4096,
    num_output=256,
    chunk_overlap_ratio=0.1,
    chunk_size_limit=None,
)

# Configuración del índice vectorial
d = 384
faiss_index = faiss.IndexFlatL2(d)
vector_store = FaissVectorStore(faiss_index=faiss_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
callback_manager = CallbackManager()

index = VectorStoreIndex.from_documents(
 documents, embed_model=Settings.embed_model, callback_manager=callback_manager
)


# Crear retriever
retriever = index.as_retriever()

# Configuración del motor de consultas
query_engine = index.as_query_engine(llm=llm)

# Consulta
response = query_engine.query("Respondeme en castellano: ¿De qué habla el texto en 20 lineas?")
#response1 = llm.complete("Respondeme en castellano: En que consiste la IA ?")

print(str(response))
