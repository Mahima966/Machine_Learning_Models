import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

# Dataset Load 
df = pd.read_csv('C:\ML_folder\Decision_Tree.csv')
 

# Sabhi features aur target column select karo
x = df.iloc[:, [1, 2, 3, 4]].values  
y = df.iloc[:, -1].values  

# Label Encoding for All Columns
label_encoders = [] 
for i in range(x.shape[1]):
    le = LabelEncoder()
    x[:, i] = le.fit_transform(x[:, i])
    label_encoders.append(le) 

# Encode target column
y_encoder = LabelEncoder()
y = y_encoder.fit_transform(y)

print(x) 

# Train model
model = GaussianNB()
model.fit(x, y)

# chack Accuracy 
accuracy = model.score(x, y)
print("Model Accuracy:", accuracy)