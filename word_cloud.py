greetings = ["hi", "hello", "hey", "helloo", "hellooo", "g morining", "gmorning", "good morning", "morning", "good day", "good afternoon", "good evening", "greetings", "greeting", "good to see you", "its good seeing you", "how are you", "how're you", "how are you doing", "how ya doin'", "how ya doin", "how is everything", "how is everything going", "how's everything going", "how is you", "how's you", "how are things", "how're things", "how is it going", "how's it going", "how's it goin'", "how's it goin", "how is life been treating you", "how's life been treating you", "how have you been", "how've you been", "what is up", "what's up", "what is cracking", "what's cracking", "what is good", "what's good", "what is happening", "what's happening", "what is new", "what's new", "what is neww", "g’day", "howdy"]

#Assumes the entire array comes from one conversation
def weightFunction(convoList, slang, freq_dict):


    weights = []
    all_words = set()
    for msg in convoList:
        for word in msg['msg']:
            all_words.add(word)
    for word in all_words:
        currScore = 3

        #Scores based on frequency in conversation
        if freq_dict.get(word, 0) < 3:
            currScore = currScore - 1
        elif freq_dict.get(word, 9) < 6:
            currScore += 2
        else:
            currScore += 3

        #Scores based on length of string
        if len(word) > 4 and len(word) < 6:
            currScore += 1
        elif len(word) > 6:
            currScore += 2
        else:
            currScore = currScore -1

        #Scores based on if string is slang
        if word in slang:
            currScore = currScore - 2
        else:
            currScore += 2
        if word in greetings:
            currScore = 0
        weights.append({
          'x': word,
          'value': currScore
        })

    return weights