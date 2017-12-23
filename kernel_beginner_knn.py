import pandas as pd
from numpy import *
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

# https://www.kaggle.com/archaeocharlie/a-beginner-s-approach-to-classification

if __name__=='__main__':
    print("Start!")
    print("Loading data......")
    labeld_images = pd.read_csv('train.csv')
    images = labeld_images.iloc[0:12000, 1:]
    labels = labeld_images.iloc[0:12000, :1]
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels,
                                                                            train_size=0.8, random_state=0)
    print("Normalaze.......")
    test_images[test_images > 0] = 1
    train_images[train_images > 0] = 1
    print("traing......")
    #clf = svm.SVC()
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(train_images, train_labels.values.ravel())
    print("testing......")
    score = clf.score(test_images, test_labels)
    print("Accuracy Score:", score)
    print("load target...")
    test_data = pd.read_csv('test.csv')
    test_data[test_data > 0] = 1
    print("predicting....")
    result = clf.predict(test_data[0:])
    print("writing result...")
    df = pd.DataFrame(result)
    df.index.name = 'ImageId'
    df.index += 1
    df.columns = ['Label']
    df.to_csv('result.csv', header=True)
    print("Done!")
