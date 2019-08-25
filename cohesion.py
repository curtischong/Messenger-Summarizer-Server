import numpy as np
def calc_cohesion(tokenized_arr, word_vecs):
  all_words = set()
  for msg in tokenized_arr:
    for word in msg['msg']:
        all_words.add(word)

  all_vecs = []
  for word in all_words:
    if(word in word_vecs):
      word_vec = word_vecs[word]
      all_vecs.append(word_vec)
  if(len(all_vecs) < 2):
    return "0"
  stacked = np.column_stack(all_vecs)
  return str(np.var(stacked))
