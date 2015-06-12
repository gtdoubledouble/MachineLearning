from sklearn.naive_bayes import GaussianNB

# training data
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])


clf = GaussianNB() # make a classifier
clf.fit(X, Y) # gives the classifier training data (features, labels)
print(clf.predict([[-0.8, -1]])) # what do you think this point is: class 1 or 2?

clf_pf = GaussianNB()
clf_pf.partial_fit(X, Y, np.unique(Y))

GaussianNB()
print(clf_pf.predict([[-0.8, -1]]))
