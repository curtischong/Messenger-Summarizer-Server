import json
from flask import Flask, request
from flask_cors import CORS
from filter import get_messages_time_blocked, get_most_important_sentence
from word_cloud import weightFunction
app = Flask(__name__)
CORS(app)

slang_file = open('slang.json')
slang_str = slang_file.read()
slang_dict = json.loads(json.loads(slang_str))


@app.route('/get_phrases', methods=['GET'])
def get_phrases():
  messages = get_messages_time_blocked(json.loads(request.args.get("messages")))
  important_sentences = []
  for convo in messages:
    if(convo == []):
      continue

    word_cloud_weight = None
    if(len(convo) > 5): # generate word cloud
      word_cloud_weight = weightFunction(convo, slang_dict)

    weight, word_freq = get_most_important_sentence(convo, slang_dict)
    important_sentences.append({
      "text": weight
      "word_cloud": word_cloud_weight
    })

  return json.dumps(important_sentences)

if __name__ == "__main__":
  app.run()