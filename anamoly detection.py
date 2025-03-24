import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score
from sklearn.linear_model import LogisticRegression
# #
# #
data = pd.read_csv(r"C:\Users\DELL\PycharmProjects\credit card\.venv\credit card.com.csv")
print(data.head())
# #
# #
print(data.info())
print(data.describe())

print(data.isnull().sum())
# # print(data.isnull().values.any())
#
# #distribution of transaction
class_count = data["Class"].value_counts()
print(class_count)
#
# # # Distribution of Valid & Fraudulent Transactions
plt.figure(figsize=(15,9))
plt.bar(class_count.index, class_count.values, color=['blue','orange'])
plt.title('Fraud vs Valid Transactions', fontsize=14)
plt.yscale('log')
plt.xlabel('Class', fontsize=12)
plt.ylabel('No. of Transactions', fontsize=12)
plt.xticks(ticks=class_count.index, labels=['Valid', 'Fraud'], fontsize=10)
plt.tight_layout()
plt.show()
#
# ## Get the Fraud and the normal dataset
fraud = data[data['Class']==1]
normal = data[data['Class']==0]
# #
print(fraud.value_counts().sum())
print(normal.value_counts().sum())
# #
print(fraud.shape,normal.shape)
# #
# # ## We need to analyze more amount of information from the transaction data
# # #How different are the amount of money used in different transaction classes?
print(fraud.Amount.describe())
# #
print(normal.Amount.describe())
# # # #
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Amount per transaction by class')
bins = 50
ax1.hist(fraud.Amount, bins = bins)
ax1.set_title('Fraud')
ax2.hist(normal.Amount, bins = bins)
ax2.set_title('Normal')
plt.xlabel('Amount ($)')
plt.ylabel('Number of Transactions')
plt.xlim((0, 20000))
plt.yscale('log')
plt.show();
# # #
plt.figure(figsize=(12,8))
plt.hist(fraud['Amount'], bins=50, color='green')
plt.title('Fraud Transactions Amount Plot', fontsize=14)
plt.yscale('log')
plt.xlabel('Amount Range', fontsize=10)
plt.ylabel('Transaction Frequency', fontsize=10)
plt.tight_layout()
plt.show()
#
plt.figure(figsize=(12,8))
plt.hist(normal['Amount'], bins=50, color='yellow')
plt.title('Valid Transactions Amount Plot', fontsize=14)
plt.yscale('log')
plt.xlabel('Amount Range', fontsize=10)
plt.ylabel('Transaction Frequency', fontsize=10)
plt.tight_layout()
plt.show()
# # #
# # # # We Will check Do fraudulent transactions occur more often during certain time frame ? Let us find out with a visual representation.
# # #
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Time of transaction vs Amount by class')
ax1.scatter(fraud.Time, fraud.Amount)
ax1.set_title('fraud')
ax2.scatter(normal.Time, normal.Amount)
ax2.set_title('normal')
plt.xlabel('Time (in Seconds)')
plt.ylabel('Amount')
plt.show()
#
#
#
# ## Correlation

import seaborn as sns
import matplotlib.pyplot as plt

# Get correlations of each feature
corrmat = data.corr()

# Select a subset of the top correlated features
top_corr_features = corrmat.index[:15]  # Adjust the number based on your dataset

# Plot heatmap for the selected features
plt.figure(figsize=(25, 25))
sns.heatmap(data[top_corr_features].corr(), annot=True, cmap="RdYlGn", fmt=".2f", annot_kws={"size": 10})
plt.tight_layout()
plt.show()
#
#
# # ## Take some sample of the data
# #
legit_sample = normal.sample(n=492,random_state=42)
print(legit_sample)
new_dataset = pd.concat([legit_sample,fraud],axis =0)
print(new_dataset)
#
# # Plot a line chart of means of each column of new_dataset
x = new_dataset[new_dataset['Class']==0].mean()
y = new_dataset[new_dataset['Class']==1].mean()
# print(x,y)
plt.figure(figsize=(12,8))
plt.plot(x, marker='o', linestyle='-', label='Valid Trans', color='blue', linewidth=2, markersize=6)
plt.plot(y, marker='o', linestyle='-', label='Fraud Trans', color='red', linewidth=2, markersize=6)
plt.title('Means of Valid & Fraud Transcations in new_dataset', fontsize=14)
plt.yscale('symlog', linthresh=1)
plt.minorticks_on()
plt.xlabel('Means of Columns')
plt.ylabel('Value of Mean')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()




#
# #checking the nature of the dataset
print(new_dataset.groupby("Class").mean())
#
# #splitting the  dataset into features and target
#
input = data.drop(columns= 'Class',axis=1)
output= data['Class']
#
print(input)
print(output)
#
# splitting the data into training and testing data

X_train, X_test , Y_train, Y_test = train_test_split(input,output,test_size=0.2,stratify=output,random_state=2)

print(input.shape, X_train.shape, X_test.shape)
#
# model training
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.svm import OneClassSVM
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


#
# Splitting the data into input (features) and output (target labels)
input = data.drop(columns='Class', axis=1)
output = data['Class']

# Splitting the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(input, output, test_size=0.2, stratify=output, random_state=2)

# Define models
# Train and evaluate Logistic Regression model
logistic_model = LogisticRegression(max_iter=10000)
logistic_model.fit(X_train, Y_train)

logistic_train_pred = logistic_model.predict(X_train)
logistic_test_pred = logistic_model.predict(X_test)

logistic_train_accuracy = accuracy_score(Y_train, logistic_train_pred)
logistic_test_accuracy = accuracy_score(Y_test, logistic_test_pred)

print(f"Logistic Regression - Train Accuracy: {logistic_train_accuracy:.4f}")
print(f"Logistic Regression - Test Accuracy: {logistic_test_accuracy:.4f}")


#
new_data = pd.read_csv(r"C:\Users\DELL\PycharmProjects\credit card\.venv\sample data credit card.csv")

new_predictions = logistic_model.predict(new_data)
print(new_predictions)

import numpy as np

unique_predictions = np.unique(new_predictions)
print("Unique predictions:", unique_predictions)


import matplotlib.pyplot as plt

plt.hist(new_predictions, bins=2, edgecolor='black')
plt.xticks([0, 1], ['Valid (0)', 'Fraud (1)'])
plt.xlabel("Prediction")
plt.ylabel("Frequency")
plt.title("Prediction Distribution")
plt.show()

# Initialize counters
count_0 = 0
count_1 = 0

for pred in new_predictions:
    if pred == 0:
        count_0 += 1
    elif pred == 1:
        count_1 += 1

print(f"Valid (0): {count_0}, Fraud (1): {count_1}")
