import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from imblearn.under_sampling import RandomUnderSampler

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#lab_frames_HR  PA-ST
#ACC_lab_windows_crop PA-ST
#lab_frames_EDA PA-ST

data_train = pd.read_csv('HRV_wmood.csv')
data_train.drop_duplicates(subset=None, keep='first', inplace=True)



data_train['2Mood'] = LabelEncoder().fit_transform(data_train['2Mood'])
label_train=data_train['2Mood']


del data_train['2Mood']



#data_train = SelectKBest(f_classif, k=3).fit_transform(data_train, label_train)

X_train, X_test, y_train, y_test = train_test_split(
    data_train, label_train, test_size=0.2, random_state=42, shuffle=False)



pipeline = Pipeline([
    ('normalizer', StandardScaler()), #Step1 - normalize data
    ('clf', LogisticRegression()) #step2 - classifier
])


pipeline.steps





clfs = []
clfs.append(SVC())
clfs.append(DecisionTreeClassifier())
clfs.append(KNeighborsClassifier())
clfs.append(RandomForestClassifier(max_depth=9))
clfs.append(LogisticRegression())

clfs.append(MLPClassifier(hidden_layer_sizes=(200,200), max_iter=400))

for classifier in clfs:
    pipeline.set_params(clf = classifier, normalizer= StandardScaler())
    scores= cross_validate(pipeline, data_train, label_train, cv=10, return_train_score=True)
    print('---------------------------------')

    print(str(classifier))

    print('---------------------------------')

    for key, values in scores.items():
        print(key, ' mean ', values.mean())
        print(key, ' std ', values.std())

    #classifier.fit(X_train, y_train)

    #y_pred = classifier.predict(X_test)

    #y_pred+y_test

    #print(classification_report(y_test, y_pred))


