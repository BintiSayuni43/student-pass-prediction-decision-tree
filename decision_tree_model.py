# ==============================================================
# STEP 1: IMPORT REQUIRED LIBRARIES
# These libraries help with data handling, machine learning,
# and visualization of the decision tree.
# ==============================================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score


# ==============================================================
# STEP 2: CREATE THE DATASET
# Here we manually create a small dataset of students
# with study hours, sleep hours, attendance, and pass result.
# ==============================================================

data = {
    "study_hours":[1,2,3,4,5,6,7,8,2,3],
    "sleep_hours":[5,6,6,7,7,8,8,7,5,6],
    "attendance":[60,65,70,75,80,85,90,95,60,68],
    "pass":[0,0,0,1,1,1,1,1,0,0]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)


# ==============================================================
# STEP 3: DATA PREPARATION
# Separate the dataset into features (X) and target (y).
# X contains the input variables used to train the model.
# y contains the output the model will predict.
# ==============================================================

X = df[["study_hours","sleep_hours","attendance"]]
y = df["pass"]

print("\nFeatures (X):")
print(X.head())

print("\nTarget (y):")
print(y.head())


# ==============================================================
# STEP 4: SPLIT THE DATA INTO TRAINING AND TESTING SETS
# Training data teaches the model.
# Testing data evaluates how well the model performs.
# ==============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==============================================================
# STEP 5: CREATE AND TRAIN THE DECISION TREE MODEL
# The model learns patterns from the training data.
# ==============================================================

model = DecisionTreeClassifier()

model.fit(X_train, y_train)


# ==============================================================
# STEP 6: MAKE PREDICTIONS
# The trained model predicts results for the test dataset.
# ==============================================================

predictions = model.predict(X_test)

print("\nPredictions:", predictions)


# ==============================================================
# STEP 7: EVALUATE MODEL PERFORMANCE
# Accuracy shows how many predictions were correct.
# ==============================================================

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", accuracy)


# ==============================================================
# STEP 8: VISUALIZE THE DECISION TREE
# This displays the decision rules used by the model.
# ==============================================================

plt.figure(figsize=(10,6))

plot_tree(
    model,
    feature_names=["study_hours","sleep_hours","attendance"],
    class_names=["Fail","Pass"],
    filled=True
)

plt.show()