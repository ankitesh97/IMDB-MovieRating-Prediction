
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.naive_bayes import GaussianNB,BernoulliNB

np.random.seed(0)

def _make_in_format():
    datadf = pd.read_csv('movie_metadata_filtered_aftercsv.csv')
    #separate classes and stuffs
    y = np.array(datadf['imdb_score'])
    datadf = datadf.drop(datadf.columns[[0,9]],axis=1)
    #normalize
    datadf = (datadf-datadf.mean())/(datadf.max()-datadf.min())
    X = np.array(datadf)

    return X,y

def Knn():
    X,y = _make_in_format()
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
    model = KNeighborsClassifier(algorithm='ball_tree')
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)
    # print y_test
    print "knn score ",accuracy_score(y_test,predictions)*100

def LogRegression():
    X,y = _make_in_format()
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)
    model = LogisticRegression(solver='newton-cg',multi_class='ovr',max_iter=200)
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)
    print "LogRegression ",accuracy_score(y_test,predictions)*100

def Svm():
    X,y = _make_in_format()
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)
    model = svm.SVC()
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)
    print "SVM ",accuracy_score(y_test,predictions)*100

def naiveBayes():
    X,y = _make_in_format()
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)
    model_guass = GaussianNB()
    model = BernoulliNB()
    model_guass.fit(X_train,y_train)
    model.fit(X_train,y_train)
    predictions_gauss = model_guass.predict(X_test)
    predictions = model.predict(X_test)
    print "naive bayes using gaussian ",accuracy_score(y_test,predictions_gauss)*100
    print "naive bayes using Bernoulli ",accuracy_score(y_test,predictions)*100




def main():
    Knn()
    LogRegression()
    Svm()
    naiveBayes()

if __name__ == '__main__':
    main()
