import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
%matplotlib inline
df = pd.read_csv(r"diabetes (1).csv") 
df.shape 
df.head(5) 
df.isnull().values.any() 
import seaborn as sns 
import matplotlib.pyplot as plt 
corrmat = df.corr() 
top_corr_features = corrmat.index 
plt.figure(figsize=(20,20)) 
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn") 
Outcome_true_count = len(df.loc[df['Outcome'] == True]) 
Outcome_false_count = len(df.loc[df['Outcome'] == False]) 
(Outcome_true_count,Outcome_false_count) 
from sklearn.model_selection import train_test_split 
feature_columns = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'] 
predicted_class = ['Outcome'] 
X = df[feature_columns].values 
y = df[predicted_class].values 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, 
random_state=10) 
from sklearn.ensemble import RandomForestClassifier 
random_forest_model = RandomForestClassifier(random_state=10) 
random_forest_model.fit(X_train, y_train.ravel()) 
predict_train_df = random_forest_model.predict(X_test) 
from sklearn import metrics 
print("Accuracy = {0:.3f}".format(metrics.accuracy_score(y_test, 
predict_train_df)))