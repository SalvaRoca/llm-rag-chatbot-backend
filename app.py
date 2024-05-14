from flask import Flask, request, abort, g
from service import langchain_service
from service import llamaindex_service

app = Flask(__name__)

rag_chain = None


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

    if rag == 'langchain':
        rag_chain = langchain_service.load_rag_chain(repo_id)
    elif rag == 'llamaindex':
        llamaindex_service.load_rag_chain(repo_id)
    else:
        abort(400, description="Invalid RAG parameter")

    return 'Model loaded: ' + llm + '/' + rag


@app.route('/queries', methods=['POST'])
def process_query():
    global rag_chain

    query = request.json['query']
    if rag_chain is None:
        abort(400, description="Model not loaded")

    return rag_chain.invoke(query)


if __name__ == '__main__':
    app.run()
