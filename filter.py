import json
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
# from spell_checker import correction

ps = PorterStemmer()

import spacy
nlp = spacy.load("en_core_web_sm")

stopwords = set(stopwords.words('english'))

slang_file = open('slang.json')
slang_str = slang_file.read()
slang_dict = json.loads(json.loads(slang_str))

def get_perc_upper(msg):
    len_msg = len(msg)
    num_upper = 0
    for char in msg:
        if char.isupper():
            num_upper += 1
    return num_upper/len_msg

#\\_()_/
# *word*
set_words = [
    'lol',
    'thicc',
    'lol',
    'lmao',
    'uwu',
    'haha',
    'rip',
    'great',
    'wtf',
    'holy',
    'beast',
    'fuck',
    'fk',
    'hehe',
    'thanks',
    'thx',
    'ahlie',
    'gtfo',
    'ikr'
]
set_words = [ps.stem(word) for word in set_words]

def has_set_word(msg):
    for word in set_words:
        if(word in msg):
            return True
    return False

def remove_based_on_rules(msg):
    if(len(msg) < 4):
        return ''
    if(get_perc_upper(msg) >= 0.5):
        return ''
    if(has_set_word(msg.lower())):
        return ''
    return msg

def get_messages(json1_data):
    messages = []
    for message in json1_data:
        data = {
            "person" : message['person'],
            "msg": message['msg']
        }
        messages.append(data)

def get_messages_time_blocked(json1_data):
    new_messages = []
    last_ts = 0
    for convo in json1_data:
        new_convos = []
        for message in convo:
            cur_text = message['msg']
            filtered = cur_text.encode('ascii', errors='ignore').decode()
            filtered = remove_based_on_rules(filtered)

            if filtered == '':
                continue

            data = {
                "person" : message['person'],
                "msg": filtered,
                "id": message['id']
            }
            new_convos.append(data)
        new_messages.append(new_convos)
    return new_messages

def fix_slang(word):
  if word in slang_dict:
    return slang_dict[word]
  return word

def simplify_messages(msgs):
    new_msgs = []
    for msg in msgs:
        new_msg = {
          'omsg': msg['omsg'],
          'msg': [],
          'id': msg['id']
        }
        for word in msg['msg']:
            word = fix_slang(word)
            # new_msg.append(ps.stem(correction(word)))
            new_msg['msg'].append(ps.stem(word))
        new_msgs.append(new_msg)
    return new_msgs

def get_tokenized_sentences(arr):
    new_msg = []
    for msg in arr:
        new_msg.append({
          'omsg': msg['msg'],
          'msg': nltk.word_tokenize(msg['msg'].lower()), # lowering is important!
          'id': msg['id']
        })
    return new_msg

def get_word_freq_dict(tokenized_arr):
    word_freq = {}
    for msg in tokenized_arr:
        for word in msg['msg']:
            if(word in word_freq):
                word_freq[word] += 1
            else:
                word_freq[word] = 0
    return word_freq

def get_most_important_sentence(convo):
    tokenized_convo = get_tokenized_sentences(convo)
    simplified_convo = simplify_messages(tokenized_convo)
    word_freq = get_word_freq_dict(simplified_convo)

    max_val = -999
    max_msg = None
    for msg in simplified_convo:
        value = 0
        for word in msg['msg']:
            value += word_freq[word.lower()]
        if(value > max_val):
            max_val = value
            max_msg = msg
    return max_msg