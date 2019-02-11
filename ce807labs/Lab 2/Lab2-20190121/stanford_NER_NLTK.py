import nltk
from nltk.tag import StanfordNERTagger
import os
path="C:/Program Files/Java/jdk1.8.0_201/bin/java.exe"
os.environ['JAVAHOME']=path
st = StanfordNERTagger('english.muc.7class.distsim.crf.ser.gz','stanford-ner.jar') 

result=st.tag('Prime Minister Theresa May travelled to Washington in 2017'.split()) 

print(result)
