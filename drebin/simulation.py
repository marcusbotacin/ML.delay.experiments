#!/usr/bin/env python
# coding: utf-8

# Malware Classification Stream Processing
# Adapted by: Marcus Botacin - 2022
# Original code: https://www.kaggle.com/datasets/fabriciojoc/fast-furious-malware-data-stream

# Import Block
import sys
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
#from sklearn.ensemble import RandomForestClassifier as Classifier
from skmultiflow.drift_detection import DDM, EDDM, ADWIN, PageHinkley
from sklearn.ensemble import RandomForestClassifier
#from skmultiflow.meta.adaptive_random_forests import AdaptiveRandomForest as Classifier
from sklearn.ensemble import RandomForestClassifier as Classifier
#from sklearn.linear_model import SGDClassifier as Classifier
import warnings
warnings.filterwarnings("ignore") # ignore deprecation warnings
import IPython # only for debugging
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score, confusion_matrix

# This class was supposed to account detection statistics
# It started complex, thus a class, but now it is very simple 
# because I discover all we need is to count accumulated FNs
# I assume FN is enough once we keep FPs below 1%
class Stats:
    def __init__(self):
        # I created a new metric (exposure)
        # it measures how users are exposed to threats
        # Either because a given number of samples is not detected 
        # Or because the same set of samples keeps being undetected for a long time
        self.exp = 0
        # total is equivalent as if no detection was made
        # the total risk one would be subject to
        self.total = 0

    # Call this function at every epoch (day, month, hour, so on)
    def set_detection(self, FNs, TPs):
            # The current exposure is the one until here plus the new FNs in this epoch
            self.exp = self.exp + FNs 
            # The total exposure
            self.total = self.total + FNs + TPs

    # Just get the statistics
    def count_exposure(self):
        ratio = 0
        # avoid division by zero
        if self.total != 0:
            # ratio is how close to full exposure we are
            # just for the case we do not want to plot absolute numbers in the future
            ratio = self.exp / float(self.total)
        return self.exp, self.total, ratio

# Oracle that always provide correct labels
# Could be a simulator of a dynamic analysis sandbox or human analyst
# in practice, all it does is to delay label delivery for a given number of epochs
class Oracle:
    def __init__(self):
        # work as a queue, storing the samples
        self.queue_samples = []
        # count the time passing
        self.queue_time = []
        # How long is the delay
        self.delay = 20

    # Model provides all data, we only store it
    def add(self, samples, predicted, true_labels):
        # Store all information
        self.queue_samples.append([samples,predicted,true_labels])
        # Store associated entry countign how many epochs remaining until data is ready
        self.queue_time.append(self.delay)

    # get detection information from the oracle
    # I'm assuming it is called once each epoch
    # if called out of context (multiple times), results will be wrong
    def get(self):
        # if list is empty, no result to deliver
        if len(self.queue_samples) != 0:
            # decrease all remaining times by one (one epoch passed)
            self.queue_time = [c-1 for c in self.queue_time]
            # get oldest element in queue (first)
            next_time = self.queue_time[0]
            next_sample = self.queue_samples[0]
            # if it is zero or less, it is ready to be delivered
            # ideally, we should check all elements in the list
            # I'm assuming only one is delivered each time
            if next_time <= 0:
                # if element will be returned, remove from the queue
                self.queue_samples = self.queue_samples[1:]
                self.queue_time = self.queue_time[1:]
                # return data
                return next_sample[0], next_sample[1], next_sample[2]
        # just for the case no data to deliver
        return [], [], []

#vectorizer class: calc average of words using word2vec
# I didn't change from Fabricio's implementation
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

# This is the model. Name according the used techniques
# The idea is to instantiate multiple of these models to run the simulation
# I moved Fabricio's implementation from inline to the class
class Word2vec_RF_DDM:
    def __init__(self):
        # The classifier itself
        self.clf = None
        # The features for word2vec or TFIDF
        self.models = []
        # The drift detector
        self.drift = DDM()
        # Save training set to include it in a future retrain
        self.Xs = []
        self.Ys = []
        # Detection threshold
        # the idea is to adjust to reach the targeted FPR
        # currently, no adjust required, FPR is always below 1%
        self.threshold = 0.51
        # The fraction of goodware samples in comparison to malware
        # Used to ensure it learns goodware better than malware
        # Thus FPR will always be low, at the cost of the detection rate
        self.ratio = 10
        # Set if dataset should be balanced according to the above ratio
        self.force_balance = True
        # Set if you want to retrain only with the data seem after a warning and not whole dataset
        self.partial_view = True
        # Instantiate oracle to produce labels
        # I integrated oracle to the model, but this was not required in fact
        # Maybe we can change it in the future
        self.oracle = Oracle()
        return

    # Balance dataset
    # I put this inside the model, not sure if the best place, but here it is
    # data_X is a pandas dataframe with all malware and goodware
    # data_Y is a numpy array with the labels
    def balance(self, data_X, data_Y):
        # first separate only the goodware samples in a new dataframe
        data_gw = data_X[data_Y==0]
        # separate the malware in a new dataframe
        data_mw = data_X[data_Y==1]
        # assuming we have more goodware than malware 
        # this was the proportion on the drebin dataset
        # if we have more malware it will likely break
        # computing the min bc in some epochs we cannot keep the proportion
        # it is important to monitor the execution
        # Anyway, we have a number of gooware that is proportional to the number of malware
        n_samples = min(self.ratio*data_mw.shape[0], data_gw.shape[0])
        # resample goodware to this number
        data_gw = data_gw.sample(n=n_samples)
        # rebuild the single dataframe with the new number of goodware plus all malware samples
        data_X = pd.concat([data_gw,data_mw])
        # create a new label array with all goodware (0) and all malware (1) labels
        data_Y = np.array(data_gw.shape[0]*[int(0)] + data_mw.shape[0]*[int(1)])
        return data_X, data_Y

    # Train a classifier
    # Retrain will also use this function
    # assume that it receives samples (x) and labels (y)
    # these vectors will be externally defined
    def train(self, X_dataTrain2,y_train2):
        # Print the number of samples in the original vectors
        print("Training request size",X_dataTrain2.shape,y_train2.shape)
        print("Training balance. [Good: %d] [Mal: %d]" % (y_train2.tolist().count(0), y_train2.tolist().count(1)))
        # Balance dataset if required
        if self.force_balance:
            X_dataTrain, y_train = self.balance(X_dataTrain2,y_train2)
        else:
            # case not required, keep the same dataset
            X_dataTrain = X_dataTrain2
            y_train = y_train2
        # Print the training dataset after rebalancing
        print("Effective Training size",X_dataTrain.shape,y_train.shape)
        print("Effective Training balance. [Good: %d] [Mal: %d]" % (y_train.tolist().count(0), y_train.tolist().count(1)))
        # Zeroing variables, new model will be trained
        self.clf = None
        self.models = []
        # if its the first time training, save training vectors for future retraining
        # No need to save every time because retraining requests will already keep track of new samples
        # We only need to append the initial training set
        if len(self.Xs) == 0:
            self.Xs = X_dataTrain
            self.Ys = y_train
        # From now, it's Fabricio's code
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
        # partial fit for the ARF case (not working well so far)
        #self.clf.partial_fit(X_train,y_train, classes=[0,1])
        self.clf.fit(X_train,y_train) #, classes=[0,1])

    # Check for concept drift
    # return if drift happens or if it is in a warning zone
    # Warning zone is important to start collect samples for retrain 
    # (in case of partial view)
    # it assumes to have access to the real labels (not true in reality)
    def check_drift(self, y_preds, y_labels):
        warning = False
        # This function is invoked every epoch
        # but drift occurs in the samples within the epoch
        # check all samples in the epoch
        for i in range(len(y_preds)):
            # add info to the drift detector
            self.drift.add_element(y_preds[i]== y_labels[i])
            # check if drift occur for that sample
            if self.drift.detected_change():
                # we have to check for every sample
                # if drift occur at any sample, then it happens for the epoch
                # if we checked only the end, a large epoch would bias the result
                # and drift detection would never happen
                # if drift happens, return immediately
                return True, False
            # if not drifting, check if close (warning)
            # do not return imediately, wait to see if drift will be detected next iter
            if self.drift.detected_warning_zone():
                warning = True
        # base case: no drift, maybe warning
        return False, warning

    # What to do when drift happen
    # hint: retrain!
    # Xs and Ys are the new vectors the new classifier will be trained with
    # the policy for establishing the vectors is externally defined
    def handle_drift(self, Xs, Ys):
        # cancel drift alert
        # in theory, if we dont, it would raise drift detection every epoch
        # in practice, it does not happen due to the large epoch bias
        self.drift.reset()
        # Default retrain policy: add new vectors to the originally training vectors
        # we might change this in the future, but it makes sense now
        # because the first training epoch is small, so we need more data
        _X = pd.concat([self.Xs,Xs])
        _Z = pd.concat([pd.DataFrame(self.Ys),Ys])
        # Ugly hack to make things compatible
        # this should be a numpy array instead a dataframe
        # so we can reuse the original training function
        _Z = _Z.to_numpy()
        _Z = _Z.reshape(_Z.shape[0],)
        # retrain
        self.train(_X,_Z)

    # Predict (Detect)
    # input: a set of samples seem in a given epoch
    def predict(self, X_dataTest):
        # It's basically Fabricio's code here
        # initialize X_train and X_test
        X_test = []
        # save models used
        models = []
        # iterate over each column of X_dataTrain
        for i in range(dataTest.values.shape[1]):
            test_data = X_dataTest.values[:,i]
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
        # if you do the following, you predict the whole vector at once
        #y_pred = self.clf.predict(X_test)
        # In the current implementation, iterate over each sample to apply custom threshold
        # The threshold is supposed to help reaching the target FPR
        y_pred = []
        # Ideally, could do at once, but there is some scikit bug here that predicts only one class
        # y_pred =[int(x[1] >= self.threshold) 
        for x in self.clf.predict_proba(X_test):
            # Thus let's use exceptions to handle the buggy 1-class
            try:
                # here successful case, compare with the threshold
                y_pred.append(int(x[1] >= self.threshold))
            except:
                # in case of error, it seems safe to assume it is a goodware (most likely)
                y_pred.append(0)
        return y_pred

# Model code ended here
# Now helper functions to create the stream

# Split a dataframe into N of a  given chunk size
# This is important to turn a single dataframe (batch) into a stream
# No balance of how many malware good in each epoch is performed
def split_dataframe(df, chunk_size = 10000): 
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
    return chunks

# get some metrics from the confusion matrix (CM)
# I was using all them in the beginning, now just FPR matters
# I wonder why these metrics are not implemented natively in scikit
# Code copied from some random stackoverflow post
def get_metrics(CM):
    TN = CM[0][0]   # Model said it was goodware and it was
    FN = CM[1][0]   # Model said it was goodware, but it was not
    TP = CM[1][1]   # Model said it was malware and it was
    FP = CM[0][1]   # Model said it was malware, but it was not
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
    # What I'm in fact using
    return FPR

# Read the input file
# Datasets are supposed to be ordered by date
# This is true for drebin and androbin datasets used in previous paper
# We need to ensure the same for new datasets as well
CSV_FILE = sys.argv[1]
data = pd.read_parquet(CSV_FILE)
# The idea here is to split the whole dataset into training and test data
# the training are all samples before a given date
initial_year = "2011-01-01"
data['submission_date'] =  pd.to_datetime(data['submission_date'])
dataTrain = data[data['submission_date'] <=initial_year]
y_train = np.array(dataTrain["label"])
# the test all samples after the date
dataTest = data[data['submission_date'] >initial_year]
y_test = np.array(dataTest["label"])
# get labels (0 = goodware, 1 = malware)
# The following is Fabricio's code
# Works well, but the decision of separating feature from label implies extra work

# remove not used columns
UNUSED_COLUMNS = ["label", "sha256", "submission_date"]
for c in UNUSED_COLUMNS:
    del dataTrain[c]
    del dataTest[c]

# Instantiate a model and train it using the training dataset
print("Training with [%d] samples to predict [%d] samples" % (dataTrain.shape[0],dataTest.shape[0]))
model = Word2vec_RF_DDM()
model.train(dataTrain,y_train)
# Call predict with the training dataset to measure the actual detection metrics
y_pred = model.predict(dataTrain)
CM = confusion_matrix(y_train,y_pred)
print("Training [Acc: %3.2f] [Prec: %3.2f] [Rec: %3.2f] [F1: %3.2f] [FPR: %3.2f] [Exp: %d]" % (100.0*accuracy_score(y_train,y_pred),100.0*precision_score(y_train,y_pred),100.0*recall_score(y_train,y_pred),100.0*f1_score(y_train,y_pred),100.0*get_metrics(CM),0))

# Split dataset in epochs
chunk_size = 500
x_batch_list = split_dataframe(dataTest,chunk_size)
y_batch_list = split_dataframe(y_test,chunk_size)

# Log files
# Need to enhance here to log all metrics
# Using argv as base name to allow running multiple instances in parallel
f =  open(sys.argv[2]+'.prec.csv','w')
f2 = open(sys.argv[2]+'.exp2.csv','w')

# Initial state, everything zeroed/empty
Y_acc = []          # no real label so far
Y_pred_acc = []     # no predicted label so far
s = Stats()         # stats class. in the current version could only an accumulator var

# the same for retraining variables
# lists of seem samples and labels
retrain_X_acc = []
retrain_Y_acc = []
retrain_Y_pred_acc = []
# controls if a warning was previously detected
prev_warning = False

# Traverse the epochs
print("Epochs = %d" % len(x_batch_list))
for epoch in range(len(x_batch_list)):
    # predict samples in current epoch
    y_pred = model.predict(x_batch_list[epoch])
    # append current window to the total sample list
    # in epoch zero, total = current window
    if epoch == 0:
        # samples
        # ideally, could be only indexes to save memory
        # one day we will have a better implementation (or no, who knows)
        X_acc = pd.DataFrame(x_batch_list[epoch])
        # actual labels in this window
        Y_acc = pd.DataFrame(y_batch_list[epoch])
        # predicted labels in this window
        Y_pred_acc = pd.DataFrame(y_pred)
    # for other epochs, append/concat
    else:
        X_acc = pd.concat([pd.DataFrame(X_acc),pd.DataFrame(x_batch_list[epoch])])
        Y_acc = pd.concat([pd.DataFrame(Y_acc),pd.DataFrame(y_batch_list[epoch])])
        Y_pred_acc = pd.concat([Y_pred_acc,pd.DataFrame(y_pred)])

    # Predicted again for all samples so far
    # This is important to check if model retrained was effective
    # Ideally, samples previously undetected are now detected
    predict_all = model.predict(X_acc)
    # Get statistics for the whole data
    # Rationale: check how exposed we are to all samples in the current epoch (day)
    CM = confusion_matrix(Y_acc,Y_pred_acc)
    # False negative is the metric that matters, as we want to detect stuff
    FN = CM[1][0]
    # True Positive is the number of actual malware detected
    TP = CM[1][1]
    # Add to the statistics. It will accumulate FNs over all days
    s.set_detection(FN, TP) 
    # Print statistics for the current epoch
    exp, total_exp, ratio = s.count_exposure()
    print("%d | [Acc: %3.2f] [Prec: %3.2f] [Rec: %3.2f] [F1: %3.2f] [FPR: %3.2f] [Exp: %d] [Texp: %d] [Rexp: %3.2f]" % (epoch,100.0*accuracy_score(Y_acc, Y_pred_acc),100.0*precision_score(Y_acc, Y_pred_acc),100.0*recall_score(Y_acc, Y_pred_acc),100.0*f1_score(Y_acc, Y_pred_acc), 100.0*get_metrics(CM), exp, total_exp, 100*ratio))
    # Print the number of samples classified. Both in total and this epoch
    # Total number does not mean correct ones. For this, print the Confusion Matrix (CM)
    # The idea of printing it here is just to have a fast way to know if it is classifying more goodware or more malware
    print("\t Local => Malware %d of %d" % (y_pred.count(1), y_batch_list[epoch].tolist().count(1)))
    print("\t Local => Goodware %d of %d" % (y_pred.count(0), y_batch_list[epoch].tolist().count(0)))
    print("\t Total => Malware: %d of %d" % (predict_all.count(1), Y_acc.value_counts()[1]))
    print("\t Total => Goodware: %d of %d" % (predict_all.count(0), Y_acc.value_counts()[0]))
    # Repeat the print but now outputing to log file
    f2.write("%d," % s.count_exposure()[0])
    f.write("%f," % (precision_score(Y_acc, Y_pred_acc)))

    # After classifying, check for drift. 
    # In case of drift, retrain
    # New model will be effective for the next epoch
    # To check for drift, we need an oracle that provides the correct labels

    # This is the ideal case, check drift known what is the actual label
    # Problem is that in real life we do not know the real label without oracle
    # We need to wait to have this label
    # We could try to speed up with partial confidence. Need to find a way to embed confidence information into the prediction
    if model.oracle is None:
        # if no oracle, just use current variables, as data are immediately available
        current_x = x_batch_list[epoch]
        current_y = y_pred
        current_true_label = y_batch_list[epoch]
    else:
        # This is the more real case
        # Current data goes to oracle
        model.oracle.add(x_batch_list[epoch],y_pred,y_batch_list[epoch])
        # And we retrieve detection label from a previous epoch
        # If oracle delay is zero, it becomes the same as the above case (immediate)
        current_x, current_y, current_true_label = model.oracle.get()

    # Check for drift or warning
    drift, warning = model.check_drift(current_y, current_true_label)
    # drift has priority
    if drift:
        # need to merge samples from the last window, before retraining
        # otherwise, we are losing important samples
        if len(retrain_X_acc) == 0:
            retrain_X_acc = pd.DataFrame(current_x)
            retrain_Y_acc = pd.DataFrame(current_true_label)
            retrain_Y_pred_acc = pd.DataFrame(current_y)
        # if not the first, append, concat
        else:
            retrain_X_acc = pd.concat([pd.DataFrame(retrain_X_acc),pd.DataFrame(current_x)])
            retrain_Y_acc = pd.concat([pd.DataFrame(retrain_Y_acc),pd.DataFrame(current_true_label)])
            retrain_Y_pred_acc = pd.concat([retrain_Y_pred_acc,pd.DataFrame(current_y)])
        # Warn users, start actual handling
        print("!!! Drift Alert !!! [Epoch: %d]" % epoch)
        # if partial view, retrain with limited view vector
        if model.partial_view:
            model.handle_drift(retrain_X_acc, retrain_Y_acc)
        # if full view allowed, retrain with all samples seem so far
        else:
            model.handle_drift(X_acc, Y_acc)
        # empty all lists, a new warning season will start
        retrain_X_acc = []
        retrain_Y_acc = []
        retrain_Y_pred_acc = []
        prev_warning = False
    # if no drift, check for warning
    # in the current window, or in a previous one
    # idea hear is to use warning to start collecting samples
    # we could be more restrictive and collect only the warning window
    elif warning or prev_warning:
        prev_warning = True
        print("+++ Drift Warning ON +++ [Epoch: %d]" % epoch)
        # if first, accumulated is itself (own window)
        if len(retrain_X_acc) == 0:
            retrain_X_acc = pd.DataFrame(current_x)
            retrain_Y_acc = pd.DataFrame(current_true_label)
            retrain_Y_pred_acc = pd.DataFrame(current_y)
        # if not the first, append, concat
        else:
            retrain_X_acc = pd.concat([pd.DataFrame(retrain_X_acc),pd.DataFrame(current_x)])
            retrain_Y_acc = pd.concat([pd.DataFrame(retrain_Y_acc),pd.DataFrame(current_true_label)])
            retrain_Y_pred_acc = pd.concat([retrain_Y_pred_acc,pd.DataFrame(current_y)])
