# Note: This spell checker takes a long time to load
# https://www.kaggle.com/cpmpml/spell-checker-using-word2vec
import gensim
import time

print("loadng wiki news")
start_time1 = time.time()
spell_model = gensim.models.KeyedVectors.load_word2vec_format('wiki-news-300d-1M.vec')
print("--- %s seconds ---" % (time.time() - start_time1))
words = spell_model.index2word
w_rank = {}
for i,word in enumerate(words):
    w_rank[word] = i
WORDS = w_rank
# Use fast text as vocabulary
def words(text): return re.findall(r'\w+', text.lower())
def P(word): 
    "Probability of `word`."
    # use inverse of rank as proxy
    # returns 0 if the word isn't in the dictionary
    return - WORDS.get(word, 0)
def correction(word): 
    "Most probable spelling correction for word."
    return max(candidates(word), key=P)
def candidates(word): 
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or [word])
def known(words): 
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)
def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)
def edits2(word): 
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))
def singlify(word):
    return "".join([letter for i,letter in enumerate(word) if i == 0 or letter != word[i-1]])

    