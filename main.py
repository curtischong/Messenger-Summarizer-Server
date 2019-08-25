import json
import time
import numpy as np
from flask import Flask, request
from flask_cors import CORS
from filter import get_messages_time_blocked, get_most_important_sentence
from word_cloud import weightFunction
from cohesion import calc_cohesion
app = Flask(__name__)
CORS(app)

slang_file = open('slang.json')
slang_str = slang_file.read()
slang_dict = json.loads(json.loads(slang_str))


USE_COHESION = True

# glove is better than wiki news
def load_vecs():
  start_time = time.time()
  print("loadng vecs")
  EMBEDDING_FILE = 'embeddings/glove.840B.300d/glove.840B.300d.txt'
  def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
  embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))
  print("--- %s seconds ---" % (time.time() - start_time))
  return embeddings_index
word_vecs = None
if USE_COHESION:
  word_vecs = load_vecs()


@app.route('/get_phrases', methods=['GET'])
def get_phrases():
  messages = get_messages_time_blocked(json.loads(request.args.get("messages")))
  important_sentences = []
  for convo in messages:
    if(convo == []):
      continue

    weight, word_freq, tokenized_arr = get_most_important_sentence(convo, slang_dict)
    word_cloud_weight = None
    if(len(convo) > 5): # generate word cloud
      word_cloud_weight = weightFunction(tokenized_arr, slang_dict, word_freq)
    cohesion_val = None
    if(USE_COHESION):
      cohesion_val = calc_cohesion(tokenized_arr, word_vecs)

    important_sentences.append({
      "text": weight,
      "word_cloud": word_cloud_weight,
      "cohesion_val": cohesion_val
    })

  return json.dumps(important_sentences)

if __name__ == "__main__":
  app.run()