from sklearn import tree

features = [[100, 1], [110, 1], [150, 0], [170, 0]]
labels = ['apple', 'apple', 'orange', 'orange']

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

print clf.predict([[160, 0], [120, 1], [90, 0]])