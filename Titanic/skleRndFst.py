# data analysis
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# https://www.kaggle.com/startupsci/titanic-data-science-solutions

if __name__=='__main__':
    print("Start!")
    print("Loading data...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    combine = [train_df, test_df]

    print("Cleanning data.")
    train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
    test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
    combine = [train_df, test_df]
    print("Cleanning data..")
    for dataset in combine:
        dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.',expand=False)
    for dataset in combine:
        dataset['Title'] = dataset['Title'].replace(['Lady','Countess','Capt','Col',\
                    'Don','Dr','Major','Rev','Sir','Jonkheer','Dona'],'Rare')
        dataset['Title'] = dataset['Title'].replace('Mlle','Miss')
        dataset['Title'] = dataset['Title'].replace('Ms','Miss')
        dataset['Title'] = dataset['Title'].replace('Mme','Mrs')
    title_mappings = {'Mr':1, "Miss":2, "Mrs":3, "Master":4, "Rare":5}
    print("Cleanning data...")
    for dataset in combine:
        dataset['Title'] = dataset['Title'].map(title_mappings)
        dataset['Title'] = dataset['Title'].fillna(0)
    train_df = train_df.drop(['Name','PassengerId'],axis=1)
    test_df = test_df.drop(['Name'], axis=1)
    combine = [train_df,test_df]
    print("Cleanning data....")
    for dataset in combine:
        dataset['Sex'] = dataset['Sex'].map({'female':1, 'male':0}).astype(int)
    for dataset in combine:
        guess_ages = np.mat(np.zeros((2,3)))
        for i in range(0,2):
            for j in range(0,3):
                guess_df = dataset[(dataset['Sex']==i)&\
                                   (dataset['Pclass']==j+1)]['Age'].dropna()
                age_guess = guess_df.median()
                guess_ages[i,j] = int(age_guess/0.5 + 0.5)*0.5
        for i in range(0,2):
            for j in range(0,3):
                dataset.loc[(dataset.Age.isnull()) & (dataset.Sex==i) & \
                            (dataset.Pclass==j+1),'Age'] = guess_ages[i,j]
        dataset['Age'] = dataset['Age'].astype(int)
    print("Cleanning data.....")
    for dataset in combine:
        dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32),'Age'] = 1
        dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
        dataset.loc[dataset['Age'] > 64, 'Age']
    for dataset in combine:
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    for dataset in combine:
        dataset['IsAlone']=0
        dataset.loc[dataset['FamilySize']==1, 'IsAlone'] = 1
    print("Cleanning data.......")
    train_df = train_df.drop(['Parch','SibSp','FamilySize'], axis=1)
    test_df = test_df.drop(['Parch','SibSp','FamilySize'],axis=1)
    combine = [train_df, test_df]
    for dataset in combine:
        dataset['Age*Class'] = dataset.Age*dataset.Pclass
    freq_port = train_df.Embarked.dropna().mode()[0]
    for dataset in combine:
        dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    for dataset in combine:
        dataset['Embarked'] = dataset['Embarked'].map({'S':0, 'C':1, 'Q':2}).astype(int)
    test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
    print("Cleanning data...................")
    for dataset in combine:
        dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
        dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
        dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
        dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
        dataset['Fare'] = dataset['Fare'].astype(int)

    X_train = train_df.drop('Survived', axis=1)
    Y_train = train_df['Survived']
    X_train_data, X_cv, Y_train_data, Y_cv = train_test_split(X_train, Y_train,train_size=0.8, random_state=0)
    X_test = test_df.drop('PassengerId',axis=1).copy()

    print("Traing......")
    # random forest
    random_forest = RandomForestClassifier(n_estimators=100)
    #random_forest = DecisionTreeClassifier()
    random_forest.fit(X_train_data, Y_train_data.values.ravel())
    score = random_forest.score(X_cv,Y_cv)
    print("Accuracy Score:", score)
    print("Predicting.....")
    Y_pred = random_forest.predict(X_test)
    submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
    print("Writing...")
    submission.to_csv('result2.csv', index=False, header=True)