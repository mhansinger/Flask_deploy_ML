# Importing the libraries

import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('hiring.csv')

X = dataset.iloc[:, :3]
y = dataset.iloc[:, -1]



model = LinearRegression()

model.fit(X, y)

# Saving model
pickle.dump(model, open('model.pkl','wb'))


# Loading model to compare the results
#model = pickle.load(open('model.pkl','rb'))
#print(model.predict([[2, 9, 6]]))  #experience, test_score, Interview Score