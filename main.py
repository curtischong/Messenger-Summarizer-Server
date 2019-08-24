import json
from flask import Flask, request
from flask_cors import CORS
from filter import get_messages_time_blocked, get_most_important_sentence
app = Flask(__name__)
CORS(app)

@app.route('/get_phrases', methods=['GET'])
def get_phrases():
  messages = get_messages_time_blocked(json.loads(request.args.get("messages")))
  important_sentences = []
  print("asdsad")
  for convo in messages:
    if(convo == []):
      continue
    important_sentences.append(get_most_important_sentence(convo))
  return json.dumps(important_sentences)

if __name__ == "__main__":
  app.run()