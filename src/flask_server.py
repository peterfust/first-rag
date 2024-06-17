from flask import Flask, request, jsonify

from src.chain import chain_with_openai_reranking, chain_with_cohere_reranking, chain_standard
from src.load_documents_parent_child import load_documents_parent_child, load_documents_standard

app = Flask(__name__)

print("Loading documents...")
retriever_parent_child = load_documents_parent_child()
retriever_standard = load_documents_standard()
print("Documents loaded!")


@app.route('/rest/text-analytics', methods=['POST'])
def invoke_chain():
    if request.method == 'POST':
        content = request.get_json()
        #response_standard = chain_standard(retriever_standard).invoke(content['question'])
        response_parent_child = chain_standard(retriever_parent_child).invoke(content['question'])
        response_rerank_cohere = chain_with_cohere_reranking(retriever_parent_child).invoke(content['question'])
        response_rerank_openai = chain_with_openai_reranking(retriever_parent_child).invoke(content['question'])
        return jsonify({
            #'response_standard': response_standard,
            'response_standard_parent_child': response_parent_child,
            'openapi_reranking': response_rerank_openai,
            'cohere_reranking': response_rerank_cohere
        }), 200


if __name__ == '__main__':
    app.run()
