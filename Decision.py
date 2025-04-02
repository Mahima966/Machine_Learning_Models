#Decision_tree_Model Using Oops
#---------------------------------#

import pandas as pd
import math


df = pd.read_csv(r"C:\ML_folder\Decision_Tree.csv")

#  values to numeric mapping
mappings = {
    'Outlook': {'Sunny': 0, 'Overcast': 1, 'Rain': 2},
    'Temp': {'Hot': 0, 'Mild': 1, 'Cool': 2},
    'Humidity': {'High': 0, 'Normal': 1},
    'Wind': {'Weak': 0, 'Strong': 1},
    'Play Tennis': {'No': 0, 'Yes': 1}
}

for col, mapping in mappings.items():
    df[col] = df[col].map(mapping)

# Define target & features
target_column = "Play Tennis"
feature_columns = ["Outlook", "Temp", "Humidity", "Wind"]

# Function to calculate entropy
def entropy(data):
    total_samples = len(data)
    label_counts = data[target_column].value_counts()
    ent = 0

    for count in label_counts:
        prob = count / total_samples
        ent -= prob * math.log2(prob)

    return ent

# Function to calculate information gain
def info_gain(data, feature):
    total_entropy = entropy(data)
    unique_values = data[feature].unique()
    weighted_entropy = 0

    for value in unique_values:
        subset = data[data[feature] == value]
        subset_entropy = entropy(subset)
        weighted_entropy += (len(subset) / len(data)) * subset_entropy

    return total_entropy - weighted_entropy

# Function to find best feature for split
def best_split(data, features):
    max_gain = -1
    best_feature = None

    for feature in features:
        gain = info_gain(data, feature)
        if gain > max_gain:
            max_gain = gain
            best_feature = feature

    return best_feature

# Function to build decision tree
def build_tree(data, features):
    labels = data[target_column].unique()

    # If only one class left, return it
    if len(labels) == 1:
        return labels[0]

    # If no more features to split
    if len(features) == 0:
        return data[target_column].mode()[0]

    # Find best feature and split data
    best_feature = best_split(data, features)

    if best_feature is None:
        return data[target_column].mode()[0]

    tree = {best_feature: {}}
    remaining_features = [f for f in features if f != best_feature]

    for value in data[best_feature].unique():
        subset = data[data[best_feature] == value]
        tree[best_feature][value] = build_tree(subset, remaining_features)

    return tree

# Build the decision tree
decision_tree = build_tree(df, feature_columns)
print("Decision Tree:", decision_tree)

# Function to make prediction
def predict(tree, input_data):
    if not isinstance(tree, dict):
        return tree

    feature = list(tree.keys())[0]  # Get first feature
    value = input_data[feature]

    if value in tree[feature]:  
        return predict(tree[feature][value], input_data)
    else:
        return 0  # Default to "No" if no matching path

# user input
user_input = {}
for feature in feature_columns:
    user_input[feature] = int(input(f"Enter {feature} ({mappings[feature]}): "))

# Make prediction
prediction = predict(decision_tree, user_input)

# result
if prediction == 1:
    print("\nYes, play tennis")
else:
    print("\nNo, do not play tennis")