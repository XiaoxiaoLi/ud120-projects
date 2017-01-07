#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

def get_ada_boost_classifier():
    from sklearn.ensemble import AdaBoostClassifier
    return AdaBoostClassifier(n_estimators=50, learning_rate=0.5)


def get_random_forest_classifier():
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(max_depth=5)


def get_knn_classifier():
    from sklearn.neighbors import NearestNeighbors
    return NearestNeighbors.KNeighborsClassifier()


def split_train_validate(features_train,labels_train):
    from sklearn.model_selection import train_test_split
    return train_test_split(features_train, labels_train, test_size = 0.3, random_state = 42)

features_train, features_validate, labels_train, labels_validate = split_train_validate(features_train,labels_train)

clf = get_random_forest_classifier()
clf = clf.fit(features_train, labels_train)

ypred = clf.predict(features_validate)

from sklearn.metrics import accuracy_score
print accuracy_score(labels_validate, ypred)

# print accuracy_score(labels_test, ypred)
#

try:
    prettyPicture(clf, features_validate, labels_validate)
except NameError:
    pass
