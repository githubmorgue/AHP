import numpy as np

x = np.array()
y = np.array()

from sklearn.svm import SVC

clf = SVC(kernel="linear").fit(x, y)
clf.fit(x, y)
print(clf.predict())