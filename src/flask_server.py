from flask import Flask, request, jsonify

from src.chain import chain_with_openai_reranking, chain_with_cohere_reranking
from src.load_documents_by_csv_parent_child import load_documents

app = Flask(__name__)

print("Loading documents...")
retriever = load_documents()
print("Documents loaded!")


@app.route('/rest/text-analytics', methods=['POST'])
def invoke_chain():
    if request.method == 'POST':
        content = request.get_json()
        response_openai = chain_with_openai_reranking(retriever).invoke(content['question'])
        response_cohere = chain_with_cohere_reranking(retriever).invoke(content['question'])
        return jsonify({'openai_reranking': response_openai, 'cohere_reranking': response_cohere}), 200


if __name__ == '__main__':
    app.run()
