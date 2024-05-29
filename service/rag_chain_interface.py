class RagChainInterface:
    def __init__(self, rag_chain):
        self.rag_chain = rag_chain

    def ask(self, query):
        if hasattr(self.rag_chain, 'query'):
            return self.rag_chain.query(query)
        elif hasattr(self.rag_chain, 'invoke'):
            return self.rag_chain.invoke({"input": query})['answer']
        else:
            raise ValueError("rag_chain has no valid method to invoke queries")
