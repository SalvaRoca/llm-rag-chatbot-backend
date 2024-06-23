import dotenv, json
from deepeval.metrics.ragas import RagasMetric
from langchain_core.prompt_values import PromptValue
from langchain_community.llms import HuggingFaceEndpoint
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics import (GEval, SummarizationMetric, AnswerRelevancyMetric, FaithfulnessMetric,
                              HallucinationMetric, BiasMetric,
                              ToxicityMetric)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams


class CustomLLM(DeepEvalBaseLLM):
    def __init__(self):
        self.load_model()
        self.model_name = "Mistral-7B-Instruct-v0.2"

    def load_model(self):
        dotenv.load_dotenv()
        self.model = HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            temperature=0.4,
        )

    def get_model_name(self):
        return self.model_name

    def generate(self, prompt):
        # Asegúrate de que el prompt sea una cadena
        if isinstance(prompt, dict):
            prompt = prompt.get("input", "")
        elif isinstance(prompt, (list, PromptValue)):
            raise ValueError("Invalid input type. Must be a string or dictionary with 'input' key.")

        try:
            response = self.model.invoke(prompt)

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

    async def a_generate(self, prompt):
        response = self.generate(prompt)
        return response


def evaluate_response(query, response, retrieval_context):
    model = CustomLLM()

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
        model=model
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
        model=model
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
        model=model
    )

    bias_metric = BiasMetric(
        model=model,
        include_reason=False
    )

    toxicity_metric = ToxicityMetric(
        model=model,
        include_reason=False
    )

    # Crear el caso de prueba
    test_case = LLMTestCase(
        input=query,
        actual_output=response,
        retrieval_context=retrieval_context,
        expected_output=""
    )

    try:
        # Evaluar el caso de prueba con las métricas
        # correctness_metric.measure(test_case)
        # clarity_metric.measure(test_case)
        # integrity_metric.measure(test_case)
        # summarization_metric.measure(test_case)
        # answer_relevancy_metric.measure(test_case)
        # faithfulness_metric.measure(test_case)
        # hallucination_metric.measure(test_case)
        bias_metric.measure(test_case)
        toxicity_metric.measure(test_case)

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
        # 'correctness_score': correctness_metric.score,
        # 'clarity_score': clarity_metric.score,
        # 'integrity_score': integrity_metric.score,
        # 'summarization_score': summarization_metric.score,
        # 'answer_relevancy_score': answer_relevancy_metric.score,
        # 'faithfulness_score': faithfulness_metric.score
        # 'hallucination_score': hallucination_metric.score,
        'bias_score': bias_metric.score,
        'toxicity_score': toxicity_metric.score
    }
