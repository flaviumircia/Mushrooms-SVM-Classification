import numpy as np
from sklearn import svm
from sklearn import model_selection
from sklearn.metrics import accuracy_score
import pandas as pd

def import_dataset():
    """
    Import the dataset in csv format
    :return: dataframe (pandas type of data)
    """
    dataframe = pd.read_csv('mushrooms.csv')
    return dataframe

def map_dataframe(val):
    """
    Maps the dataframe String values in integers necessary for the SVM algorithm
    :param val: value of the String
    :return: A dictionary (hash map) with integer values (corresponding to each val letter)
    """
    if val in category:
        return category[val]
    else:
        category[val]=len(category)
    return category[val]

def split_the_dataframe(df):
    """
    Splits the dataframe in train and test, then splits the train in data and labels (the same for the test dataset)
    :param df: initial undivided dataframe
    :return: train data, train labels, test data and test labels
    """
    train,test=model_selection.train_test_split(df,test_size=0.25)
    train_labels=train['class']
    test_labels=test['class']
    train.drop('class',axis=1,inplace=True)
    test.drop('class',axis=1,inplace=True)
    return train,train_labels,test,test_labels

if __name__ == '__main__':
    """
    Main function. Calling the other methods
    """
    # creating the dataframe
    df=import_dataset()

    # calling the hash map function for the dataframe
    for i in range(df.shape[1]): # for i in 0, n columns
        category = {} # initial empty dictionary
        df.iloc[:, i] = df.iloc[:, i].apply(map_dataframe) # indexing the dataframe columns by indices,
                                                            # then applying the mapping function on an axis

    #getting the divided data from the splitting method
    train_data,train_labels,test_data,test_labels=split_the_dataframe(df)

    #iterates through the power necessary for the cost function (from -5 to 7). The stop power (last one ) not included!
    for i in np.arange(-5,8,1,dtype=float):
        cost=np.power(2,i) #calculates the cost

        model=svm.SVC(kernel='linear',C=cost) #setting the model with linear kernel and different cost value
        model.fit(train_data,train_labels)  #fitting the model with train data and labels

        predicted_labels=model.predict(test_data) # predicting the labels using the test data

        print("Accuracy = ",accuracy_score(test_labels,predicted_labels),", the cost =",cost,", kernel = linear")
        # getting the accuracy score based on comparing each test label with each predicted label
        # the accuracy score is the total of correct result (test_label[n]==predicted_label[n]) out of the total number of labels
