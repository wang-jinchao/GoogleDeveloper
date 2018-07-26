import random
from scipy.spatial import distance

def euc(a,b):
    return distance.euclidean(a,b)

# print('===(3,4)',euc((1,2),(4,6)))
class Wang():
    def fit(self,X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train

    def predict(self,X_test):
        predictions = []
        for row in X_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions
    def closest(self,row):
        best_distance = euc(row,self.X_train[0])
        best_indx = 0
        for i in range(1, len(self.X_train)):
            distance = euc(row,self.X_train[i])
            if best_distance >distance:
                best_distance = distance
                best_indx = i
        return self.Y_train[best_indx]

from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
Y = iris.target

from sklearn.model_selection import train_test_split
from sklearn import ensemble

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5)

# from sklearn import tree
# my_classfier = tree.DecisionTreeClassifier()
# from sklearn.neighbors import KNeighborsClassifier
# my_classfier = KNeighborsClassifier()
my_classfier = Wang()

my_classfier.fit(X_train, Y_train)
predictions = my_classfier.predict(X_test)
print(type(predictions),'====')
print('===')
print(predictions[0],'===')
print(predictions)
print(Y_test)
print(predictions - Y_test)

# clf = ensemble.RandomForestClassifier()
# clf.fit(X_train, Y_train)
# predictions = clf.predict(X_test)
# print("Using Random Forest Classifier, Predictions are:")
# print(predictions)

from sklearn.metrics import accuracy_score

print("Accuracy Score in percent is:")
# score = accuracy_score(predictions , Y_test)
score = accuracy_score(Y_test, predictions)
print(score * 100)