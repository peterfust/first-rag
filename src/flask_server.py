from flask import Flask, request, jsonify
from src.load_documents_by_csv_parent_child import load_documents
from src.chain import chain

app = Flask(__name__)

# Add your initialization code here
print("Loading documents...")
retriever = load_documents()
print("Documents loaded!")


@app.route('/rest/text-analytics', methods=['POST'])
def invoke_chain():
    if request.method == 'POST':
        content = request.get_json()
        response = chain(retriever).invoke(content['question'])
        return jsonify({'llm_response': response}), 200


if __name__ == '__main__':
    app.run()
