import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import json
import numpy as np


def get_hot_idx(arr):
    max_val = -1
    max_idx = -1
    for idx in range(0, len(arr)):
        e = arr[idx]
        if max_val < e:
            max_val = e
            max_idx = idx
    return max_idx


# Process data & train
data = pd.read_csv("./data/mnist_train.csv")

X = data.iloc[:, 1:]
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

rfc = RandomForestClassifier(n_jobs=-1, n_estimators=10)
rfc.fit(X_train, y_train)
score = rfc.score(X_test, y_test)
'''0.9404'''

# apply to my data
with open('./data/mnist_png_testing/images_not_normal.json') as data_file:
    data = json.load(data_file)
unknown = np.zeros((len(data), 784))
for i in range(len(data)):
    unknown[i] = data[i]

y_out = rfc.predict(unknown)

with open('./data/mnist_png_testing/correctValues.json') as data_file:
    data = json.load(data_file)
correct_vals = []
for d in data:
    correct_val = get_hot_idx(d)
    correct_vals.append(correct_val)

correct_times = 0
for i in range(0, len(y_out)):
    if y_out[i] == correct_vals[i]:
        correct_times = correct_times + 1

test_score = correct_times / len(y_out)

'''0.9428'''
