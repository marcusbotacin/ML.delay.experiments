#!/usr/bin/env python
# coding: utf-8
import sys
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
#from sklearn.ensemble import RandomForestClassifier as Classifier
from skmultiflow.drift_detection import DDM, EDDM, ADWIN, PageHinkley
from sklearn.ensemble import RandomForestClassifier
from skmultiflow.meta.adaptive_random_forests import AdaptiveRandomForest as Classifier
#from sklearn.ensemble import RandomForestClassifier as Classifier
#from sklearn.linear_model import SGDClassifier as Classifier
import warnings
warnings.filterwarnings("ignore")
import IPython
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score, confusion_matrix

class Stats:
    def __init__(self):
        self.exp = 0
    def set_detection(self, detected, should_detect):
        self.exp = self.exp + (should_detect - detected)

    def count_exposure(self):
        return self.exp

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


class Word2vec_RF_DDM:
    def __init__(self):
        self.clf = None
        self.models = []
        self.drift = DDM()
        self.Xs = []
        self.Ys = []
        self.threshold = 0.51
        return

    def balance(self, data_X, data_Y):
        #print(type(data_X),type(data_Y))

        #dfy = pd.DataFrame(data_Y)
        #mask = dfy[0]==0
        #data_mw = data_X[mask.reindex(data_X.index, fill_value=False)]
        #mask = dfy[0]==1
        #data_gw = data_X[mask.reindex(data_X.index, fill_value=False)]
        #data_mw = data_mw.sample(n=data_gw.shape[0])
        #data_X = pd.concat([data_mw,data_gw])
        #data_Y = np.array(data_gw.shape[0]*[int(0)] + data_gw.shape[0]*[int(1)])

        #print(type(data_X),type(data_Y))
        return data_X, data_Y

    def train(self, X_dataTrain2,y_train2):
        #print("Received to train:",type(X_dataTrain2),type(y_train2))
        print("Training request size",X_dataTrain2.shape,y_train2.shape)
        print("Training balance. [Good: %d] [Mal: %d]" % (y_train2.tolist().count(0), y_train2.tolist().count(1)))
        X_dataTrain, y_train = self.balance(X_dataTrain2,y_train2)
        self.clf = None
        self.models = []
        #print("Training...")
        print("Effective Training size",X_dataTrain.shape,y_train.shape)
        #print(type(X_dataTrain),type(y_train))
        #print(X_dataTrain.shape,y_train.shape)
        self.Xs = X_dataTrain
        self.Ys = y_train
        # initialize X_train and X_test
        X_train = []
        # iterate over each column of X_dataTrain
        for i in range(X_dataTrain.values.shape[1]):
            # train feature word2vec using column i
            print("\tTraining column '{}' ({}) ...".format(data.columns.values[i], i))
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
        print("\tTraining Random Forest")
        # initialize classifier
        self.clf = Classifier(random_state=0)
        # train classifier
        self.clf.partial_fit(X_train,y_train, classes=[0,1])

    def check_drift(self, y_preds, y_labels):
        #print("Check drift")
        for i in range(len(y_preds)):
            self.drift.add_element(y_preds[i]== y_labels[i])
            if self.drift.detected_change():
                return True
        return False

    def handle_drift(self, Xs, Ys):
        self.drift.reset()
        _X = pd.concat([self.Xs,Xs])
        _Z = pd.concat([pd.DataFrame(self.Ys),Ys])
        _Z = _Z.to_numpy()
        _Z = _Z.reshape(_Z.shape[0],)
        self.train(_X,_Z)

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
        #y_pred = self.clf.predict(X_test)
        y_pred = []
        # y_pred =[int(x[1] >= self.threshold) 
        for x in self.clf.predict_proba(X_test):
            try:
                y_pred.append(int(x[1] >= self.threshold))
            except:
                y_pred.append(0)
                #IPython.embed()
        # y_pred =[int(x[1] >= self.threshold) for x in self.clf.predict_proba(X_test)]
        #print(self.clf.predict(X_test)[0])
        #print(self.clf.predict_proba(X_test)[0])
        #print(y_pred[0])
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
print("Training with [%d] samples to predict [%d] samples" % (dataTrain.shape[0],dataTest.shape[0]))
model = Word2vec_RF_DDM()
model.train(dataTrain,y_train)

y_pred = model.predict(dataTrain)

CM = confusion_matrix(y_train,y_pred)
#print(CM)
#print("Training FPR: %f" % get_metrics(CM))
print("Training [Acc: %3.2f] [Prec: %3.2f] [Rec: %3.2f] [F1: %3.2f] [FPR: %3.2f] [Exp: %d]" % (100.0*accuracy_score(y_train,y_pred),100.0*precision_score(y_train,y_pred),100.0*recall_score(y_train,y_pred),100.0*f1_score(y_train,y_pred),100.0*get_metrics(CM),0))

# Split dataset in epochs
chunk_size = 500
#x_batch_list = np.array_split(dataTest,chunk_size)
#y_batch_list = np.array_split(y_test,chunk_size)
x_batch_list = split_dataframe(dataTest,chunk_size)
y_batch_list = split_dataframe(y_test,chunk_size)

# Traverse the epochs
f = open('prec.csv','w')
f2 = open('exp.csv','w')

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
        X_acc = pd.DataFrame(x_batch_list[epoch])
        Y_acc = pd.DataFrame(y_batch_list[epoch])
        Y_pred_acc = pd.DataFrame(y_pred)
    else:
        #print(type(Y_acc),type(y_batch_list[epoch]))
        X_acc = pd.concat([pd.DataFrame(X_acc),pd.DataFrame(x_batch_list[epoch])])
        Y_acc = pd.concat([pd.DataFrame(Y_acc),pd.DataFrame(y_batch_list[epoch])])
        Y_pred_acc = pd.concat([Y_pred_acc,pd.DataFrame(y_pred)])

    # This is the ideal case, check drift known what is the actual label
    # Problem is that in real life we do not know the real label without oracle
    # We need to wait to have this label
    # We could try to speed up with partial confidence. Need to find a way to embed confidence information into the prediction
    if model.check_drift(y_pred, y_batch_list[epoch]):
        print("!!! Drift Alert !!! [Epoch: %d]" % epoch)
        #model.handle_drift(X_acc, Y_acc)
    
    #s.set_detection(X_acc,Y_acc,Y_pred_acc)


    ## Predicted again
    predict_all = model.predict(X_acc)


    s.set_detection(predict_all.count(1), Y_acc.value_counts()[1])
    #s.set_detection(x_batch_list[epoch], y_pred, y_batch_list[epoch])
    #s.new_epoch()

    #print("Exposure = %d" % s.count_exposure())
#y_test = model.predict(dataTest)
#print(y_test)
    #print("%d: [F1 Acc=%f]" % (epoch,f1_score(Y_acc, Y_pred_acc)))
    CM = confusion_matrix(Y_acc,Y_pred_acc)
    #print("Testing FPR: %f" % get_metrics(CM))

    print("%d | [Acc: %3.2f] [Prec: %3.2f] [Rec: %3.2f] [F1: %3.2f] [FPR: %3.2f] [Exp: %d]" % (epoch,100.0*accuracy_score(Y_acc, Y_pred_acc),100.0*precision_score(Y_acc, Y_pred_acc),100.0*recall_score(Y_acc, Y_pred_acc),100.0*f1_score(Y_acc, Y_pred_acc), 100.0*get_metrics(CM), s.count_exposure()))

    print("\t Local => Malware %d of %d" % (y_pred.count(1), y_batch_list[epoch].tolist().count(1)))
    print("\t Local => Goodware %d of %d" % (y_pred.count(0), y_batch_list[epoch].tolist().count(0)))
    print("\t Total => Malware: %d of %d" % (predict_all.count(1), Y_acc.value_counts()[1]))
    print("\t Total => Goodware: %d of %d" % (predict_all.count(0), Y_acc.value_counts()[0]))
    #print("%d: [Prec: %f]" % (epoch,precision_score(Y_acc, Y_pred_acc)))
    f2.write("%d," % s.count_exposure())
    f.write("%f," % (precision_score(Y_acc, Y_pred_acc)))
    #print("%d: [F1=%f]" % (epoch,f1_score(y_batch_list[epoch], y_pred)))
    #f.write("%f\n" % f1_score(y_batch_list[epoch],y_pred))
    #print(confusion_matrix(Y_acc, Y_pred_acc))
#IPython.embed()
