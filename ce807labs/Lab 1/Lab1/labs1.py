Python 3.6.6 (v3.6.6:4cf1f54eb7, Jun 27 2018, 03:37:03) [MSC v.1900 64 bit (AMD64)] on win32
Type "copyright", "credits" or "license()" for more information.
>>> import sklearn
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> data = np.array([[1,2], [2,3], [3,4], [4,5], [5,6]])
>>> x = data[:,0]
>>> y= data[:,1]
>>> x
array([1, 2, 3, 4, 5])
>>> y
array([2, 3, 4, 5, 6])
>>> plt.scatter(x,y)
<matplotlib.collections.PathCollection object at 0x000000000D931080>
>>> plt.grid(True)
>>> plt.show()
>>> from sklearn.feature_extraction.text import CountVectorizer
>>> vectorizer = CountVectorizer(min_df=1)
>>> content = ["How to format my hard disk", " Hard disk format problems "]
>>> X = vectorizer.fit_transform(content)
>>> vectorizer.get_feature_names()
['disk', 'format', 'hard', 'how', 'my', 'problems', 'to']
>>> X.toarray()
array([[1, 1, 1, 1, 1, 0, 1],
       [1, 1, 1, 0, 0, 1, 0]], dtype=int64)
>>> X.toarray()[0]
array([1, 1, 1, 1, 1, 0, 1], dtype=int64)
>>> X.toarray()[1,2]
1
>>> from sklearn.datasets import fetch_20newsgroups
>>> categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
>>> twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
>>> from sklearn.feature_extraction.text import CountVectorizer
>>> vectorizer = CountVectorizer()
>>> train_counts = vectorizer.fit_transform(twenty_train.data)
>>> vectorizer.vocabulary_.get('algorithm')
4690
>>> len(vectorizer.get_feature_names())
35788
>>> vectorizer = CountVectorizer(stop_words='english')
>>> sorted(vectorizer.get_stop_words())[:20]
['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'all', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'am', 'among', 'amongst', 'amoungst']
>>> import nltk
>>> s = nltk.stem.SnowballStemmer('english')
>>> s.stem("cats")
'cat'
>>> nltk.download()
showing info https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/index.xml
True
>>> from nltk.tokenize import word_tokenize
>>> text = word_tokenize("And now for something completely different")
>>> text
['And', 'now', 'for', 'something', 'completely', 'different']
>>> nltk.pos_tag(text)
[('And', 'CC'), ('now', 'RB'), ('for', 'IN'), ('something', 'NN'), ('completely', 'RB'), ('different', 'JJ')]
>>> from sklearn.feature_extraction.text import CountVectorizer
>>> vectorizer = CountVectorizer(stop_words='english')
>>> analyze = vectorizer.build_analyzer()
>>> analyze("John bought carrots and potatoes")
['john', 'bought', 'carrots', 'potatoes']
>>> 
KeyboardInterrupt
>>> import nltk.stem
>>> english_stemmer = nltk.stem.SnowballStemmer(“English”)
SyntaxError: invalid character in identifier
>>> english_stemmer = nltk.stem.SnowballStemmer(“english”)
SyntaxError: invalid character in identifier
>>> english_stemmer = nltk.stem.SnowballStemmer('english')
>>> class StemmedCountVectorizer(CountVectorizer):
	def build_analyzer(self):
		analyzer = super(StemmedCountVectorizer, self).build_analyzer()
		return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))

	
>>> stem_vectorizer = StemmedCountVectorizer(min_df=1, stop_words=”english”)
SyntaxError: invalid character in identifier
>>> stem_vectorizer = StemmedCountVectorizer(min_df=1, stop_words='english')
>>> stem_analyze = stem_vectorizer.build_analyzer()
>>> Y = stem_analyze("John bought carrots and potatoes")
>>> Y
<generator object StemmedCountVectorizer.build_analyzer.<locals>.<lambda>.<locals>.<genexpr> at 0x00000000197812B0>
>>> for tok in Y:
	print(tok)

	
john
bought
carrot
potato
>>> from sklearn.datasets import fetch_20newsgroups
>>> categories = ['alt.atheism','soc.religion.christian',
'comp.graphics', 'sci.med']
>>> twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
>>> train_counts = stem_vectorizer.fit_transform(twenty_train.data)
>>> len(stem_vectorizer.get_feature_names())
26888
>>> 
