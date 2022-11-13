#!/usr/bin/env python
# coding: utf-8
import sys
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier as Classifier
import IPython
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score, confusion_matrix

class Stats:
    def __init__(self):
        self.last_detection = dict()
        self.count_detection = dict()
        self.sample_ground = dict()
    def set_detection(self,samples,detections,groundtruth):
        for s,d, g in zip(samples,detections, groundtruth):
            self.sample_ground[s] = g
            self.last_detection[s] = d
            if s not in self.count_detection:
                self.count_detection[s] = 0
    def new_epoch(self):
        for s in self.last_detection:
            if self.last_detection[s] != self.sample_ground[s]:
                self.count_detection[s] = self.count_detection[s]+1
    def count_exposure(self):
        exp = 0
        for s in self.count_detection:
            exp = exp + self.count_detection[s]
        return exp

#vectorizer class: calc average of words using word2vec
class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(list(word2vec.values())[0])

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


class Word2vec_RF_noupdate:
    def __init__(self):
        self.clf = None
        self.models = []
        return

    def train(self, X_dataTrain,y_train):
        # initialize X_train and X_test
        X_train = []
        # iterate over each column of X_dataTrain
        for i in range(X_dataTrain.values.shape[1]):
            # train feature word2vec using column i
            print("Training column '{}' ({}) ...".format(data.columns.values[i], i))
            # get train and test data
            train_data = X_dataTrain.values[:,i]
            # initialize and train model
            w2v = Word2Vec(train_data,vector_size=100,min_count=1,sg=0)
            # get generated space
            space = dict(zip(w2v.wv.index_to_key,w2v.wv.vectors))
            # initialize vectorizer
            m = MeanEmbeddingVectorizer(space)
            m.fit(train_data,y_train)
            # transform train and test texts to w2v mean
            train_w2v = m.transform(train_data)
            # if first execution, save only features
            if len(X_train) == 0:
                X_train = train_w2v
            # concatenate existing features
            else:
                X_train = np.concatenate((X_train, train_w2v), axis=1)
            # save model
            self.models.append(m)
        print("Training Random Forest")
        # initialize classifier
        self.clf = Classifier()
        # train classifier
        self.clf.fit(X_train,y_train)

    def predict(self, X_dataTest):
        # initialize X_train and X_test
        X_test = []
        # save models used
        models = []
        # iterate over each column of X_dataTrain
        for i in range(dataTest.values.shape[1]):
            test_data = X_dataTest.values[:,i]
            # initialize and train model
            #w2v = Word2Vec(train_data,vector_size=100,min_count=1,sg=0)
            # get generated space
            #space = dict(zip(w2v.wv.index_to_key,w2v.wv.vectors))
            # initialize vectorizer
            #m = MeanEmbeddingVectorizer(space)
            # transform train and test texts to w2v mean
            m = self.models[i] 
            test_w2v = m.transform(test_data)
            # if first execution, save only features
            if len(X_test) == 0:
                X_test = test_w2v
            # concatenate existing features
            else:
                X_test = np.concatenate((X_test, test_w2v), axis=1)
            # save model
            models.append(m)
        # predict test data
        y_pred = self.clf.predict(X_test)
        return y_pred

def split_dataframe(df, chunk_size = 10000): 
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
    return chunks

def get_metrics(CM):
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)

    return FPR

# This is code to handle androbin dataset
CSV_FILE = sys.argv[1]
data = pd.read_parquet(CSV_FILE)

initial_year = "2011-01-01"
data['submission_date'] =  pd.to_datetime(data['submission_date'])
dataTrain = data[data['submission_date'] <=initial_year]
y_train = np.array(dataTrain["label"])

dataTest = data[data['submission_date'] >initial_year]
y_test = np.array(dataTest["label"])

# get labels (0 = goodware, 1 = malware)
#labels = data['label']
# remove not used columns
UNUSED_COLUMNS = ["label", "sha256", "submission_date"]
for c in UNUSED_COLUMNS:
    del dataTrain[c]
    del dataTest[c]

#IPython.embed()
# Train
print(dataTrain.size,dataTest.size)
model = Word2vec_RF_noupdate()
model.train(dataTrain,y_train)

y_pred = model.predict(dataTrain)
print("Training %f" % f1_score(y_train,y_pred))
CM = confusion_matrix(y_train,y_pred)
print(CM)
print("FPR %f" % get_metrics(CM))

# Split dataset in epochs
chunk_size = 500
#x_batch_list = np.array_split(dataTest,chunk_size)
#y_batch_list = np.array_split(y_test,chunk_size)
x_batch_list = split_dataframe(dataTest,chunk_size)
y_batch_list = split_dataframe(y_test,chunk_size)

# Traverse the epochs
f = open('res.txt','w')

Y_acc = []
Y_pred_acc = []
print("Epochs = %d" % len(x_batch_list))
s = Stats()
for epoch in range(len(x_batch_list)):
    #print("Chunk size %d" % len(x_batch_list[epoch]))
    y_pred = model.predict(x_batch_list[epoch])
    #print("%d: [Acc=%f]" % (epoch,accuracy_score(y_batch_list[epoch], y_pred)))
    #print("%d: [Pre=%f]" % (epoch,precision_score(y_batch_list[epoch], y_pred)))
    #print("%d: [Rec=%f]" % (epoch,recall_score(y_batch_list[epoch], y_pred)))
    #print("%d: [F1=%f]" % (epoch,f1_score(y_batch_list[epoch], y_pred)))
    #print(confusion_matrix(y_batch_list[epoch], y_pred))
    if epoch == 0:
        Y_acc = pd.DataFrame(y_batch_list[epoch])
        Y_pred_acc = pd.DataFrame(y_pred)
    else:
        #print(type(Y_acc),type(y_batch_list[epoch]))
        Y_acc = pd.concat([pd.DataFrame(Y_acc),pd.DataFrame(y_batch_list[epoch])])
        Y_pred_acc = pd.concat([Y_pred_acc,pd.DataFrame(y_pred)])


    s.set_detection(x_batch_list[epoch],y_batch_list[epoch],y_pred)
    s.new_epoch()
    print("Exposure = %d" % s.count_exposure())
#y_test = model.predict(dataTest)
#print(y_test)
    #print("%d: [F1 Acc=%f]" % (epoch,f1_score(Y_acc, Y_pred_acc)))
    print("%d: [Prec Acc=%f]" % (epoch,precision_score(Y_acc, Y_pred_acc)))
    f.write("%d," % s.count_exposure())
    #f.write("%f," % (precision_score(Y_acc, Y_pred_acc)))
    #print("%d: [F1=%f]" % (epoch,f1_score(y_batch_list[epoch], y_pred)))
    #f.write("%f\n" % f1_score(y_batch_list[epoch],y_pred))
    #print(confusion_matrix(Y_acc, Y_pred_acc))
#IPython.embed()
