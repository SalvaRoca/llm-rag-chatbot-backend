import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from service import langchain_service
from service import llamaindex_service

app = Flask(__name__)
cors = CORS(app, resources={
    r"/*": {
        "origins": [
            "*"
        ],
        "methods": ["GET", "POST", "OPTIONS", "HEAD"],
        "allow_headers": ["Content-Type"]
    }
})

llm = ""
rag = ""
rag_chain = None


@app.route('/models', methods=['POST'])
def load_model():
    if 'files' not in request.files:
        return jsonify({'error': 'No files loaded'}), 400
    files = request.files.getlist('files')
    for file in files:
        if file.filename == '':
            return jsonify({'error': 'No file found'}), 400
        if file.content_type != 'application/pdf':
            return jsonify({'error': 'Invalid file type, only PDF format is supported'}), 400
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.root_path, 'data', filename))
        print('File uploaded: ' + filename)

    global rag_chain
    global llm
    global rag

    llm = request.args.get('llm')
    rag = request.args.get('rag')

    if llm != 'mistral' and llm != 'llama':
        return jsonify({'error': 'Invalid LLM parameter'}), 400
    if rag == 'langchain':
        rag_chain = langchain_service.load_rag_chain(llm)
    elif rag == 'llamaindex':
        rag_chain = llamaindex_service.load_rag_chain(llm)
    else:
        return jsonify({'error': 'Invalid RAG parameter'}), 400
    for filename in os.listdir('data'):
        file_path = os.path.join('data', filename)
        if filename != ".gitkeep":
            try:
                os.unlink(file_path)
            except Exception as e:
                print(e)
    return 'Model loaded: ' + llm + '/' + rag


@app.route('/models', methods=['GET'])
def get_model():
    global llm
    global rag
    if llm == "" or rag == "":
        return jsonify({'error': 'Model not loaded'}), 400
    return jsonify({"llm": llm, "rag": rag})


@app.route('/queries', methods=['POST'])
def process_query():
    global rag_chain

    if 'query' not in request.json or 'messages' not in request.json:
        return jsonify({'error': 'Invalid request format: query and messages are required.'}), 400
    query = request.json['query']
    messages = request.json['messages']

    if rag_chain is not None:
        response = rag_chain.ask(query, messages)
    else:
        return jsonify({'error': 'Model not loaded'}), 400
    return response


if __name__ == '__main__':
    app.run()
