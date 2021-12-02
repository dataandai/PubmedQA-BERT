from flask import Flask, request, jsonify
from flask_cors import CORS
import backend.squad as squad
import backend.entity_recognition as entity_recognition
import backend.pubmed_parser as pubmed_parser
#import backend.wiki_parser

app = Flask(__name__)
CORS(app)


@app.route("/qa", methods=['POST'])
def predict_qa():

    q = request.args.get("question")

    docEntities = entity_recognition.recognize_entities(q)
    doc = pubmed_parser.parse_entity(docEntities)

    try:
        return jsonify({"result": squad.get_answer(q, doc)})
    except Exception as e:
        print(e)
        return jsonify({"result": "Model Failed"})


@app.route("/", methods=['GET'])
def index():
    return "api: /qa?question=..."


if __name__ == "__main__":
    app.run('0.0.0.0', port=8000)
