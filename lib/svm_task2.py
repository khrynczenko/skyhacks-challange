import pandas as pd
from sklearn import svm, tree, ensemble, naive_bayes
from sklearn.model_selection import KFold
from sklearn import multiclass
import pickle
import numpy as np
from joblib import dump, load

df = pd.read_csv("/Users/szymek/Downloads/processed_thresholded.csv")
df = df.drop(['filename'], axis=1)
df = df[df["task2_class"] != "validation"]

df["task2_class"] = df["task2_class"].replace("house", 0)
df["task2_class"] = df["task2_class"].replace("dining_room", 1)
df["task2_class"] = df["task2_class"].replace("kitchen", 2)
df["task2_class"] = df["task2_class"].replace("bathroom", 3)
df["task2_class"] = df["task2_class"].replace("living_room", 4)
df["task2_class"] = df["task2_class"].replace("bedroom", 5)
df = df.reset_index()
kf = KFold(n_splits=10, random_state=42, shuffle=True)
kf.get_n_splits(df)
print(kf)

avg = []
for train_index, test_index in kf.split(df):
    X_train, X_test = df.loc[train_index].drop(["task2_class", "index"], axis=1), df.loc[test_index].drop(
        ["task2_class", "index"], axis=1)
    y_train, y_test = df.loc[train_index, "task2_class"], df.loc[test_index, "task2_class"]
    clf = svm.SVC()
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))
    avg.append(clf.score(X_test, y_test))
print(f"AVG:{np.mean(avg)}")

clf = svm.SVC()
clf.fit(df.drop(["task2_class", "index"], axis=1), df["task2_class"])

dump(clf, 'SVM_2_thresholded.joblib')
