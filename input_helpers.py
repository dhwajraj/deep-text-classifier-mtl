import numpy as np
import re
import itertools
from collections import Counter
import numpy as np
import time
import data_helpers
import gc
from gensim.models.word2vec import Word2Vec
import gzip


class InputHelper(object):
    pre_emb = dict()
    
    def loadW2V(self,type="bin"):
        print("Loading W2V data...")
        if type=="text":
            # this seems faster than gensim non-binary load
            for line in gzip.open('/home/ubuntu/GoogleNews-vectors-modified.txt.gz'):
                l = line.strip().split()
                pre_emb[l[0]]=np.asarray(l[1:])
        else:
            pre_emb = Word2Vec.load_word2vec_format('/home/ubuntu/GoogleNews-vectors-negative300.bin',binary=True)
            pre_emb.init_sims(replace=True)
        print("loaded word2vec len ", len(pre_emb))
        gc.collect()
    
    def getTsvData(filepath):
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
    
    
    # Data Preparatopn
    # ==================================================
    
    
    def getDataSets(self, training_paths, max_document_length, filter_h_pad, percent_dev):
        x_list=[]
        y_list=[]
        for i in xrange(multi_train_size):
            x_temp,y_temp = getTsvData("/home/ubuntu/all_training/"+training_paths[i])
            x_list.append(x_temp)
            y_list.append(y_temp)
            del x_temp
            del y_temp
        
        # Build vocabulary
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length-filter_h_pad,min_frequency=2)
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
            del x
            del x_text
            del y
            # Split train/test set
            # TODO: This is very crude, should use cross-validation
            dev_idx = -1*len(y_shuffled)//percent_dev
            x_train, x_dev = x_shuffled[:dev_idx], x_shuffled[dev_idx:]
            y_train, y_dev = y_shuffled[:dev_idx], y_shuffled[dev_idx:]
            print("Train/Dev split for {}: {:d}/{:d}".format(training_paths[i1], len(y_train), len(y_dev)))
            sum_no_of_batches = sum_no_of_batches+(len(y_train)//FLAGS.batch_size)
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
        return train_set,dev_set