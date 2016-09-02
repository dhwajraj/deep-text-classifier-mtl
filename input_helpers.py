import numpy as np
import re
import itertools
from collections import Counter
import numpy as np
import time
import data_helpers
import gc
from tensorflow.contrib import learn
from gensim.models.word2vec import Word2Vec
import gzip


class InputHelper(object):
    pre_emb = dict()
    
    def loadW2V(self,emb_path, type="textgz"):
        print("Loading W2V data...")
	num_keys = 0
        if type=="textgz":
            # this seems faster than gensim non-binary load
            for line in gzip.open(emb_path):
                l = line.strip().split()
                self.pre_emb[l[0]]=np.asarray(l[1:])
		num_keys=len(self.pre_emb)
        else:
            self.pre_emb = Word2Vec.load_word2vec_format(emb_path,binary=True)
            self.pre_emb.init_sims(replace=True)
            num_keys=len(self.pre_emb.vocab)
        print("loaded word2vec len ", num_keys)
        gc.collect()

    def deletePreEmb(self):
        self.pre_emb=dict()
        gc.collect()
    
    def getTsvData(self, filepath):
        print("Loading training data from "+filepath)
        x=[]
        y=[]
        for line in open(filepath):
            l=line.strip().split("\t")
            if len(l)<2:
                continue
            x.append(l[1])
            v=np.array([0,1])
            if l[0]=="-1" or l[0]=="0":
                v=np.array([1,0])
            y.append(v)
        return np.asarray(x),np.asarray(y)
    
    def dumpValidation(self,x_text,y,shuffled_index,dev_idx,i):
	print("dumping validation "+str(i))
	x_shuffled=x_text[shuffled_index]
	y_shuffled=y[shuffled_index]
	x_dev=x_shuffled[dev_idx:]
        y_dev=y_shuffled[dev_idx:]
	del x_shuffled
	del y_shuffled
	with open('validation.txt'+str(i),'w') as f:
	    for text,label in zip(x_dev,y_dev):
		f.write(str(label)+"\t"+text+"\n")
	    f.close()
	del x_dev
	del y_dev
	
    # Data Preparatopn
    # ==================================================
    
    
    def getDataSets(self, training_paths, max_document_length, filter_h_pad, percent_dev, batch_size):
        x_list=[]
        y_list=[]
	multi_train_size = len(training_paths)
        for i in xrange(multi_train_size):
            x_temp,y_temp = self.getTsvData(training_paths[i])
            x_list.append(x_temp)
            y_list.append(y_temp)
            del x_temp
            del y_temp
        # Build vocabulary
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length-filter_h_pad,min_frequency=1)
        vocab_processor.fit_transform(np.concatenate(x_list,axis=0))
        print len(vocab_processor.vocabulary_)
        i1=0
        train_set=[]
        dev_set=[]
        sum_no_of_batches = 0
        for x_text,y in zip(x_list, y_list):
            x = np.asarray(list(vocab_processor.transform(x_text)))
            x = np.concatenate((np.zeros((len(x),filter_h_pad)),x),axis=1)
            # Randomly shuffle data
            np.random.seed(10)
            shuffle_indices = np.random.permutation(np.arange(len(y)))
            x_shuffled = x[shuffle_indices]
            y_shuffled = y[shuffle_indices]
            dev_idx = -1*len(y_shuffled)//percent_dev
            self.dumpValidation(x_text,y,shuffle_indices,dev_idx,i1)
            del x
            del x_text
            del y
            # Split train/test set
            # TODO: This is very crude, should use cross-validation
            x_train, x_dev = x_shuffled[:dev_idx], x_shuffled[dev_idx:]
            y_train, y_dev = y_shuffled[:dev_idx], y_shuffled[dev_idx:]
            print("Train/Dev split for {}: {:d}/{:d}".format(training_paths[i1], len(y_train), len(y_dev)))
            sum_no_of_batches = sum_no_of_batches+(len(y_train)//batch_size)
            train_set.append((x_train,y_train))
            dev_set.append((x_dev,y_dev))
            del x_shuffled
            del y_shuffled
            del x_train
            del x_dev
            i1=i1+1
        del x_list
        del y_list
        gc.collect()
        return train_set,dev_set,vocab_processor,sum_no_of_batches
