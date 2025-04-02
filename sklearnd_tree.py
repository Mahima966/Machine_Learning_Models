import pandas as pd
from sklearn.tree import DecisionTreeClassifier,plot_tree
import matplotlib.pyplot as plt

df=pd.read_csv('C:\ML_folder\Decision_Tree.csv')


df['Outlook'] = df['Outlook'].map({'Sunny': 0, 'Overcast': 1, 'Rain': 2})
df['Temp'] = df['Temp'].map({'Hot': 0, 'Mild': 1, 'Cool': 2})
df['Humidity'] = df['Humidity'].map({'High': 0, 'Normal': 1})
df['Wind'] = df['Wind'].map({'Weak': 0, 'Strong': 1})
df['Play Tennis'] = df['Play Tennis'].map({'No': 0, 'Yes': 1})

# Decision Tree Classifier
X = df[['Outlook', 'Temp', 'Humidity', 'Wind']]
y = df['Play Tennis']

clf = DecisionTreeClassifier()
clf.fit(X, y)

plt.figure(figsize=(8, 4))
plot_tree(clf, feature_names=X.columns, class_names=['No', 'Yes'], filled=True)
plt.show()


# Prediction
outlook = int(input("Enter outlook (0=Sunny, 1=Overcast, 2=Rain): "))
temp = int(input("Enter temperature (0=Hot, 1=Mild, 2=Cool): "))
humidity = int(input("Enter humidity (0=High, 1=Normal): "))
wind = int(input("Enter wind (0=Weak, 1=Strong): "))


# Manually calculate prediction
if clf.tree_.feature[0] == 0:  # Outlook
    if outlook == 0:  # Sunny
        if clf.tree_.feature[1] == 1:  # Temp
            if temp == 0:  # Hot
                 prediction = clf.tree_.value[2][0][1]  # Yes
            else:
                prediction = clf.tree_.value[2][0][0]  # No
        else:
            prediction = clf.tree_.value[1][0][1]  # Yes
    elif outlook == 1:  # Overcast
        prediction = clf.tree_.value[1][0][1]  # Yes
    else:  # Rain
        if clf.tree_.feature[1] == 2:  # Humidity
            if humidity == 0:  # High
                prediction = clf.tree_.value[2][0][0]  # No
            else:
                prediction = clf.tree_.value[2][0][1]  # Yes
        else:
            prediction = clf.tree_.value[1][0][0]  # No
else:
    prediction = clf.tree_.value[0][0][0]  # No

if prediction == 1:
    print("Yes, play tennis")
else:
    print("No, do not play tennis")


