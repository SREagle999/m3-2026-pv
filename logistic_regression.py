# example with a multinomial logistic regression on the iris dataset
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
# from sklearn.datasets import load_iris   # classification - iris dataset straight from sklearn

# Data prep (Label_encoding for ordinal data instead of one-hot encoding)
df = pd.read_csv('datasets/iris/bezdekIris.data', sep=",") # check with print(df.head())
x = df.iloc[:,0:4].values # test with print(x)
y = df.iloc[:,4].values # test with print(y)
le = LabelEncoder()
y = le.fit_transform(y) # check with print(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# LogisticRegression
logReg = LogisticRegression(max_iter=200, solver='lbfgs')   # now automatically uses multinomial code
logReg.fit(x_train, y_train)
y_pred = logReg.predict(x_test)

# Evaluation
print("Accuracy Score: " + '{:.2}'.format(accuracy_score(y_test, y_pred)))
print("Simple CM Display:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:\n"+classification_report(y_test, y_pred))

cmLabels = ['Setosa','Versicolor','Virginica'] # hard-coded labels for a confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", yticklabels=cmLabels, xticklabels=cmLabels) # annot indicates presence of values on squares, fmt is format of numbers
plt.title("Confusion Matrix")
plt.show()