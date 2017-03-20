# IMDB-MovieRating-Prediction
A machine learning application to predict imdb moview rating based on some features.

## About the data set
The data set along with its description is defined [here](https://www.kaggle.com/deepmatrix/imdb-5000-movie-dataset).  

## Steps
* After preprocessing/cleaning the data there were around 2000 data points.    
* The main task was to predict the IMDB rating of a movie.  
* This was considered as a classification problem by taking 10 classes 1-10 i.e the rating.  
* There were initially many features which was then reduced using the domain knowledge finally only 9 features was taken into consideration, the filtered and processed data is saved in the after csv.csv file.  
* All the models are pickled in the models folder.  

## Usage
To view the predictions run the main.py just change the filename of your data  

## Results

Results are summarized in the table  
* First, the accuracy was calculated by letting the model only predict the exact rating.


| Model  	| Accuracy (%) 	|  
|--------------------	|--------------	|  
| K Nearest Neighbours| 37.52       	|  
| Logitic Regression  | 40.9         	|  
| SVC                 | 36.35        	|  
| Naive Bayes(Bernoulli)| 33.67       |  

* Secondly, the accuracy was calculated by letting the model only predict the range of rating i.e with the error of +-1, so for e.g if the rating predicted was 8 then the accuracy was tested if the actual label was between 8-1 to 8+1.

| Model  	| Accuracy (%) 	|  
|--------------------	|--------------	|  
| K Nearest Neighbours| 80.90      	|  
| Logitic Regression  | 85.09        |  
| SVC                 | 83.91        	|  
| Naive Bayes(Bernoulli)| 80.23      |  


## Conclusion
Finally, the Conclusion that I made from the results    
* Predicting the exact rating is quiet difficult also any predictions wouldn't be that accurate because the main contribution towards the rating is the story of the movie.  
* There seems very less relation between the stats and the rating since some actors may be good but the movie's facebook likes wouldn't be that high because of less publicity of the movie.    
* L2 regularized logistic Regression gave the best results as compared to other classification algorithms.
