import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
#from sklearn.externals import joblib

df = pd.read_csv('data/diabetes.csv')

x = df.drop(['Outcome'],1)
y = df['Outcome']

x_train, x_text, y_train, y_test = train_test_split(x, y, test_size=0.2)

logistic_regression = LogisticRegression()
logistic_regression.fit(x_train, y_train)

accuracy = logistic_regression.score(x_train, y_train)
print("accuracy = ", accuracy * 100, "%")

coeff = list(logistic_regression.coef_[0])
labels = list(x_train.columns)
features = pd.DataFrame()
features['Features'] = labels
features['importance'] = coeff
features.sort_values(by=['importance'], ascending=True, inplace=True)
features['positive'] = features['importance'] > 0
features.set_index('Features', inplace=True)
features.importance.plot(kind='barh', figsize=(11, 6),color = features.positive.map({True: 'blue', False: 'red'}))
plt.xlabel('Importance')