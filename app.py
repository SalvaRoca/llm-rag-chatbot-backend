import os
from flask import Flask, request, abort
from flask_cors import CORS
from werkzeug.utils import secure_filename
from service import langchain_service
from service import llamaindex_service

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

langchain_rag = None
llamaindex_rag = None
ultima_conversacion = None
memory = []

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        print('No file part')
        abort(400, description="No file part")
    files = request.files.getlist('files')
    for file in files:
        if file.filename == '':
            print('No selected file')
            abort(400, description="No selected file")
        if file.content_type != 'application/pdf':
            print('Invalid file type, only PDFs are allowed')
            abort(400, description="Invalid file type, only PDFs are allowed")
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.root_path, 'data', filename))
        print('File uploaded: ' + filename)
    return "Files uploaded successfully", 200

@app.route('/models', methods=['POST'])
def load_model():
    global langchain_rag, llamaindex_rag
    llm = request.args.get('llm')
    rag = request.args.get('rag')
    if llm == 'mistral':
        repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
    elif llm == 'llama':
        repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    else:
        abort(400, description="Invalid LLM parameter")
    if rag == 'langchain':
        langchain_rag = langchain_service.load_rag_chain(repo_id)
    elif rag == 'llamaindex':
        llamaindex_rag = llamaindex_service.load_rag_chain(repo_id)
    else:
        abort(400, description="Invalid RAG parameter")
    return 'Model loaded: ' + llm + '/' + rag

@app.route('/queries', methods=['POST'])
def process_query():
    global langchain_rag, llamaindex_rag, user_input, bot_response, ultima_conversacion
    query = request.json['query']
    if memory:
        ultima_conversacion = memory[-1]
    if ultima_conversacion is not None:
        user_input = ultima_conversacion["pregunta"]
        bot_response = ultima_conversacion["respuesta"]
        query = f"Respóndeme solo en español: {query}"
    if llamaindex_rag is not None:
        response = llamaindex_rag.query(
            "Respóndeme solo en español, si no tiene que ver con los datos aportados, pues con la informaión que sepas y en español las líneas que necesites: " + query)
    elif langchain_rag is not None:
        response = langchain_rag.invoke(query)
    else:
        abort(400, description="Model not loaded")
    response_str = str(response)
    memory.append({"pregunta": query, "respuesta": response})
    return response_str

if __name__ == '__main__':
    app.run()
