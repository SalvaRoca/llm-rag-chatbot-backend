from flask import Flask, request, abort, g
from service import langchain_service
from service import llamaindex_service

app = Flask(__name__)

rag_chain = None
query_engine =None
memory = []
@app.route('/models', methods=['POST'])
def load_model():
    llm = request.args.get('llm')
    rag = request.args.get('rag')

    if llm == 'mistral':
        repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
    elif llm == 'llama':
        repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    else:
        abort(400, description="Invalid LLM parameter")

    global rag_chain
    global query_engine
    if rag == 'langchain':
        rag_chain = langchain_service.load_rag_chain(repo_id)
    elif rag == 'llamaindex':
        query_engine = llamaindex_service.load_rag_chain(repo_id)
    else:
        abort(400, description="Invalid RAG parameter")

    return 'Model loaded: ' + llm + '/' + rag





@app.route('/queries', methods=['POST'])
def process_query():
    global rag_chain
    global query_engine
    query = request.json['query']
    if rag_chain is not None:
        response = process_query(query, query_engine)
        return response
    if query_engine is not None:
        #response = query_engine.query("Respóndeme, si no tiene que ver con los datos aportados, pues con la informaión que sepas y en español: " + query)
        response = process_query(query, query_engine)
        response_str = str(response)
        return response_str
    abort(400, description="Model not loaded")

def save_to_memory(user_input, bot_response):
    """Guarda la conversación en la memoria."""
    memory.append({"pregunta": user_input, "respuesta": bot_response})
def get_last_conversation():
    """Obtiene la última conversación de la memoria."""
    if memory:
        return memory[-1]
    return None
def process_query(query, query_engine):
    global rag_chain
    """Procesa la consulta del usuario y utiliza la memoria para proporcionar contexto."""
    last_conversation = get_last_conversation()
    if last_conversation:
        user_input = last_conversation["pregunta"]
        bot_response = last_conversation["respuesta"]
        # Agregar contexto a la consulta actual
        query = f"{bot_response}\n\n{query}"

    # Procesar la consulta
    if query_engine is not None:
        response = query_engine.query("Respóndeme solo en español, si no tiene que ver con los datos aportados, pues con la informaión que sepas y en español: " + query)

    else:
        response = rag_chain.invoke(query)
    # Guardar la conversación actual en la memoria
    response_str = str(response)

    save_to_memory(query, response_str)

    return response_str

if __name__ == '__main__':
    app.run()
