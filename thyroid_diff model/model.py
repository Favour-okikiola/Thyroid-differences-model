import numpy as np
import pandas as pd
import matplotlib 
import matplotlib.pyplot as plt
import seaborn as sns

Data = pd.read_csv("C:\Thyroid_Diff.csv")
Data = Data.rename(columns = {'Hx Radiothreapy':'Hx_Radiothreapy', 'Thyroid Function':'Thyroid_Function', 'Physical Examination':'Physical_Examination'})
Data.drop(columns=['Hx Smoking','T','N','M'], inplace=True)
Data.head()

# dropping duplicated values
Data = Data.drop_duplicates(keep='first')
# spam_detection.duplicated().value_counts()
# Data.head()

# shape of the dataset
Data.shape

X = Data.drop(columns=["Recurred"])  #feature
y = Data["Recurred"] #target
# print(X.value_counts())
# print(y.value_counts())

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder 
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# preprocessing target
encoder = LabelEncoder()
y = encoder.fit_transform(y)
# y

# splitting into train and test set
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size=0.8, random_state=42)



# preprocessing  features
num_features = X.select_dtypes(include=['int64']).columns
cat_features = X.select_dtypes(include=['object']).columns

num_transformer = StandardScaler()
cat_transformer = OneHotEncoder(handle_unknown="ignore")
# cat_transformer = LabelEncoder()

preprocessor = ColumnTransformer(transformers=[('num', num_transformer, num_features),('cat',cat_transformer, cat_features)])

# creating pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),('regression', LogisticRegression(random_state=42))])

# training
pipeline.fit(Xtrain,ytrain)

# prediction
y_pred = pipeline.predict(Xtest)
# print('prediction:', y_pred)
# print('actual:', ytest)

score = pipeline.score(Xtrain, ytrain)
print('score:', score)

report = classification_report(ytest, y_pred)
print(report)

from sklearn.metrics import confusion_matrix
import seaborn as sns
mat = confusion_matrix(ytest, y_pred)
sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('predicted value')
plt.ylabel('true value');

# making pickle file
import pickle
with open("model.pkl", "wb") as f:
    pickle.dump(pipeline, f)
print('model saved as model.pkl')