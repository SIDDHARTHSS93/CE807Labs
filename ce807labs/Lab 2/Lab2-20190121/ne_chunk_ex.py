import nltk
import re
import time

exampleArray = ['Prime Minister:Theresa May travelled to Switzerland in 2018.']


##let the fun begin!##
def nerInNLTK():
    try:
        for item in exampleArray:
            tokenized = nltk.word_tokenize(item)
            pos_tagged = nltk.pos_tag(tokenized)
            print (pos_tagged)

            namedEnt = nltk.ne_chunk(pos_tagged)
            namedEnt.draw()

            time.sleep(1)

    except Exception as e:
        print(e)

nerInNLTK()
