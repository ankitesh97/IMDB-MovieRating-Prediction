
import pickle
import pandas as pd
from preprocess import process
from sklearn.linear_model import LogisticRegression

FILENAME = 'movie_metadata_filtered_aftercsv.csv'

def main():
    model = pickle.loads(open('models/LogRegression_thre1'))
    # provide your filename here
    process(filename='your_file_name.csv')
    datadf = pd.read_csv(FILENAME)
    datadf = datadf.drop(datadf.columns[[0]],axis=1)
    datadf = (datadf-datadf.mean())/(datadf.max()-datadf.min())
    X = np.array(datadf)
    predictions = model.predict(X)
    # consider the predicted rating to be in the range of +.- 1
    # for example of predicted is 7 then it may be between 6-8
    print predictions

if __name__ == '__main__':
    main()
