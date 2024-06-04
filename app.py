import os
from flask import Flask, request, abort
from flask_cors import CORS
from werkzeug.utils import secure_filename
from service import langchain_service
from service import llamaindex_service

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

rag_chain = None

@app.route('/models', methods=['POST'])
def load_model():
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

    global rag_chain
    llm = request.args.get('llm')
    rag = request.args.get('rag')
    if llm == 'mistral':
        repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
    elif llm == 'llama':
        repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    else:
        abort(400, description="Invalid LLM parameter")
    if rag == 'langchain':
        rag_chain = langchain_service.load_rag_chain(repo_id)
    elif rag == 'llamaindex':
        rag_chain = llamaindex_service.load_rag_chain(repo_id)
    else:
        abort(400, description="Invalid RAG parameter")
    for filename in os.listdir('data'):
        file_path = os.path.join('data', filename)
        try:
            os.unlink(file_path)
        except Exception as e:
            print(e)
    return 'Model loaded: ' + llm + '/' + rag


@app.route('/queries', methods=['POST'])
def process_query():
    global rag_chain
    query = request.json['query']
    if rag_chain is not None:
        response = rag_chain.ask(query)
    else:
        abort(400, description="Model not loaded")
    return str(response)

if __name__ == '__main__':
    app.run()
