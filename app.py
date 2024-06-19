import asyncio
import json
import os
from flask import Flask, request, abort, jsonify
from flask_cors import CORS
from langchain_core.prompt_values import PromptValue
from werkzeug.utils import secure_filename
from deepeval.metrics import GEval, SummarizationMetric
from deepeval.test_case import LLMTestCaseParams
from service import langchain_service
from service import llamaindex_service
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

def evaluate_response(query, response, custom_llm):
    # Definir las métricas
    correctness_metric = GEval(
        name="Correctness",
        criteria="Determine whether the actual output is factually correct based on the expected output.",
        evaluation_steps=[
            "Check whether the facts in 'actual output' contradict any facts in 'expected output'",
            "You should also heavily penalize omission of detail",
            "Vague language, or contradicting OPINIONS, are OK"
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        model=custom_llm
    )

    clarity_metric = GEval(
        name="Clarity",
        criteria="Evaluate if the actual output is clear and easy to understand.",
        evaluation_steps=[
            "Check if the language used in the actual output is simple and straightforward.",
            "Ensure that the response avoids jargon and complex terms unless necessary.",
            "The response should be concise and to the point."
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        model=custom_llm
    )

    integrity_metric = GEval(
        name="Integrity",
        criteria="Evaluate if the actual output covers all aspects of the expected answer comprehensively.",
        evaluation_steps=[
            "Ensure that the response addresses all parts of the query.",
            "Check if the response includes all relevant information needed to fully answer the question.",
            "Penalize omissions of critical details."
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        model=custom_llm
    )
    summarization_metric = SummarizationMetric(
        model=custom_llm
    )

    answer_relevancy_metric = AnswerRelevancyMetric(
        model=custom_llm
    )

    # Crear el caso de prueba
    test_case = LLMTestCase(
        input=query,
        actual_output=response,
        expected_output=""  # Este sería normalmente la respuesta correcta
    )

    try:
        # Evaluar el caso de prueba con las métricas
        correctness_metric.measure(test_case)
        clarity_metric.measure(test_case)
        integrity_metric.measure(test_case)
        summarization_metric.measure(test_case)
        answer_relevancy_metric.measure(test_case)
    except json.JSONDecodeError as e:
        return {
            'error': 'JSONDecodeError',
            'message': str(e)
        }
    except ValueError as e:
        return {
            'error': 'ValueError',
            'message': str(e)
        }

    return {
        'correctness_score': correctness_metric.score,
        'correctness_reason': correctness_metric.reason,
        'clarity_score': clarity_metric.score,
        'clarity_reason': clarity_metric.reason,
        'integrity_score': integrity_metric.score,
        'integrity_reason': integrity_metric.reason,
        'summarization_score': summarization_metric.score,
        'summarization_reason': summarization_metric.reason,
        'answer_relevancy_score': answer_relevancy_metric.score,
        'answer_relevancy_reason': answer_relevancy_metric.reason
    }


class CustomLLM(DeepEvalBaseLLM):
    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name

    def generate(self, prompt, **kwargs):
        # Asegúrate de que el prompt sea una cadena
        if isinstance(prompt, dict):
            prompt = prompt.get("input", "")
        elif isinstance(prompt, (list, PromptValue)):
            raise ValueError("Invalid input type. Must be a string or dictionary with 'input' key.")

        try:
            response = self.model.invoke(prompt)
            # Depuración: Imprimir la respuesta del modelo para verificar su formato
            print("Response from model:", response)

            # Verificar si la respuesta es un dict y contiene 'output' o si es una cadena
            if isinstance(response, dict):
                if 'output' in response:
                    return response['output']
                elif 'score' in response and 'reason' in response:
                    return response
                else:
                    raise ValueError(f"Invalid response format from model: {response}")
            elif isinstance(response, str):
                return response
            else:
                raise ValueError(f"Invalid response type from model: {type(response)}")
        except Exception as e:
            raise ValueError(f"Error generating response: {e}")

    async def a_generate(self, prompt, **kwargs):
        response = self.generate(prompt, **kwargs)
        return response

    def get_model_name(self):
        return self.model_name

    def load_model(self, model_name):
        self.model = load_model_llm(model_name)


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

rag_chain = None
responses = []
custom_model = None
custom_model_name = None  # Añadido para almacenar el nombre del modelo

def load_model_llm(model_name):
    if model_name == 'mistral':
        return langchain_service.load_model("mistralai/Mistral-7B-Instruct-v0.2")
    elif model_name == 'llama':
        return langchain_service.load_model("meta-llama/Meta-Llama-3-8B-Instruct")
    else:
        raise ValueError("Invalid model name")

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

    global rag_chain, custom_model, custom_model_name
    llm = request.args.get('llm')
    rag = request.args.get('rag')
    try:
        custom_model = load_model_llm(llm)
        custom_model_name = llm  # Almacenamos el nombre del modelo
    except ValueError as e:
        abort(400, description=str(e))

    if rag == 'langchain':
        rag_chain = langchain_service.load_rag_chain(custom_model)
    elif rag == 'llamaindex':
        rag_chain = llamaindex_service.load_rag_chain(custom_model)
    else:
        abort(400, description="Invalid RAG parameter")

    for filename in os.listdir('data'):
        file_path = os.path.join('data', filename)
        if filename != ".gitkeep":
            try:
                os.unlink(file_path)
            except Exception as e:
                print(e)
    return 'Model loaded: ' + llm + '/' + rag

@app.route('/queries', methods=['POST'])
def process_query():
    global rag_chain, responses
    query = request.json['query']
    if rag_chain is not None:
        response = rag_chain.ask(query)
        responses.append({'query': query, 'response': response})

        # Crear una instancia de CustomLLM con el modelo cargado
        custom_llm = CustomLLM(custom_model, custom_model_name)

        # Llamar a la función de evaluación
        evaluation_results = evaluate_response(query, response, custom_llm)

        return jsonify({
            'query': query,
            'response': response,
            **evaluation_results
        })

    else:
        abort(400, description="Model not loaded")
   # return jsonify({'query': query, 'response': response})


if __name__ == '__main__':
    app.run()
