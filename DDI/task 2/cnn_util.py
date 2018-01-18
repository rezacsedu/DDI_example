import pickle
from gensim.models import Word2Vec
import numpy as np
from sklearn.cross_validation import train_test_split
from collections import Counter

def get_data(fname = "features.pickle"):
	f = open(fname,'rb')
	data = pickle.load(f)
	f.close()
	return data


def get_idx_from_tokens(tokens, word_idx_map, max_l=51, k=50, filter_h=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = filter_h - 1
    for i in xrange(pad):
        x.append(0)
    
    for word in tokens:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l+2*pad:
        x.append(0)
    return x

def make_idx_data_cv(data, k=50, filter_h=5,PE = False):
	"""
	Function:Transforms sentences into a 2-d matrix.
	"""
	# first, build vocabulary, index
	model = Word2Vec.load_word2vec_format('we_embedding', binary=False)
	vocabs = model.index2word
	vocab_size = len(vocabs)
	word_idx_map = dict()
	W = np.zeros(shape=(vocab_size+1, k), dtype='float32')            
	W[0] = np.zeros(k, dtype='float32')
	i = 1
	for word in vocabs:
	    W[i] = model[word]
	    word_idx_map[word] = i
	    i += 1

	#index labels and get the maximum length
	labels = []
	length = []
	for each in data:
		label = each[2][2]
		labels.append(label)
		length.append(len(each[1]))
	max_l = max(length)
	labels = Counter(labels).most_common()
	label2idx = dict()
	i = 0
	for each in labels:
		label = each[0]
		label2idx[label] = i
		i += 1
	#{u'int': 4, u'advise': 3, u'false': 0, u'effect': 1, u'mechanism': 2}
	#build training and testing set
	all_data = []
	for each in data:
		tokens = each[1]
		sent = get_idx_from_tokens(tokens, word_idx_map, max_l, k, filter_h)  
		# add position embedfing and label 
		pe = each[2][:2]
		label = each[2][2]
		if PE:
			sent.extend(pe)
		sent.append(label2idx[label]) 
		all_data.append(sent)   

	all_data = np.array(all_data,dtype="int")
	return all_data, W, max_l
	

def split_train_test(all_data):
	labels = []
	for each in all_data:
		labels.append(each[-1])
	X_train, X_test, y_train, y_test = train_test_split(all_data, labels, test_size=0.33, random_state=42)
	return [X_train,X_test]