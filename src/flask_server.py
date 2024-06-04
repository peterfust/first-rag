from flask import Flask, request, jsonify

from src.chain import chain
from src.load_documents_by_csv_parent_child import load_documents

app = Flask(__name__)

print("Loading documents...")
retriever = load_documents()
print("Documents loaded!")


@app.route('/rest/text-analytics', methods=['POST'])
def invoke_chain():
    if request.method == 'POST':
        content = request.get_json()
        question = content['question']
        response = chain(retriever).invoke(question)
        print(response)
        return jsonify({'llm_response': response}), 200


if __name__ == '__main__':
    app.run()
