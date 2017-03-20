
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.naive_bayes import GaussianNB,BernoulliNB
import pickle

np.random.seed(0)

FILENAME = 'movie_metadata_filtered_aftercsv.csv'
THRESHOLD_PREDICTION = 1

def _make_in_format(filename):
    datadf = pd.read_csv(filename)
    #separate classes and stuffs
    y = np.array(datadf['imdb_score'])
    datadf = datadf.drop(datadf.columns[[0,9]],axis=1)
    #normalize
    datadf = (datadf-datadf.mean())/(datadf.max()-datadf.min())
    X = np.array(datadf)

    return X,y

def _pickle_it(model,filename):
    a = pickle.dumps(model)
    write_file = open('models/'+filename,'w')
    write_file.write(a)

def accuracy_score(y_test,predictions):
        correct = []
        for i in range(len(y_test)):
            if y_test[i]>=predictions[i]-THRESHOLD_PREDICTION and y_test[i]<=predictions[i]+THRESHOLD_PREDICTION:
                correct.append(1)
            else:
                correct.append(0)

        accuracy = sum(map(int,correct))*1.0/len(correct)
        return accuracy

def Knn():
    X,y = _make_in_format(FILENAME)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
    model = KNeighborsClassifier(algorithm='ball_tree')
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)
    # print y_test
    _pickle_it(model,"Knn_thre1")
    print "knn score ",accuracy_score(y_test,predictions)*100

def LogRegression():
    X,y = _make_in_format(FILENAME)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)
    model = LogisticRegression(solver='newton-cg',multi_class='ovr',max_iter=200,penalty='l2')
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)
    _pickle_it(model,"LogRegression_thre1")
    print "LogRegression ",accuracy_score(y_test,predictions)*100

def Svm():
    X,y = _make_in_format(FILENAME)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)
    model = svm.SVC()
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)
    _pickle_it(model,"svm_thre1")
    print "SVM ",accuracy_score(y_test,predictions)*100

def naiveBayes():
    X,y = _make_in_format(FILENAME)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)
    model_guass = GaussianNB()
    model = BernoulliNB()
    model_guass.fit(X_train,y_train)
    model.fit(X_train,y_train)
    predictions_gauss = model_guass.predict(X_test)
    predictions = model.predict(X_test)
    _pickle_it(model,"Bernoulli_thre1")
    _pickle_it(model_guass,"guass_thre1")
    print "naive bayes using gaussian ",accuracy_score(y_test,predictions_gauss)*100
    print "naive bayes using Bernoulli ",accuracy_score(y_test,predictions)*100




def main():
    Knn()
    LogRegression()
    Svm()
    naiveBayes()

if __name__ == '__main__':
    main()
