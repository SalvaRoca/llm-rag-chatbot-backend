from .deepeval_service import evaluate_response


class RagChainInterface:
    def __init__(self, rag_chain):
        self.rag_chain = rag_chain

    def ask(self, query, messages):
        if hasattr(self.rag_chain, 'query'):
            history = "\n".join([f"{msg['author']}: {msg['text']}" for msg in messages])
            query = f"""
                Y teniendo en cuenta el historial de la conversación en el que yo soy el Usuario: {history}
                Responde en español a la siguiente petición. Si la pregunta no está relacionada con el contexto o la 
                respuesta a la misma no se encuentra en el contexto, respóndela con tus propios conocimientos ignorando 
                el contexto: {query}
                Responde a partir de aquí en español y usando notación Markdown:
                """
            response_object = self.rag_chain.query(query)
            answer = response_object.response
            retrieval_context = [node.get_content() for node in response_object.source_nodes]
            evaluation_results = evaluate_response(query, answer, retrieval_context)
            print(evaluation_results)
            return answer

        elif hasattr(self.rag_chain, 'invoke'):
            response = self.rag_chain.invoke({"input": query, "history": messages})
            answer = response['answer']
            retrieval_context = [doc.page_content for doc in response['context']]
            evaluation_results = evaluate_response(query, answer, retrieval_context)
            print(evaluation_results)
            return answer

        else:
            raise ValueError("rag_chain has no valid method to invoke queries")
