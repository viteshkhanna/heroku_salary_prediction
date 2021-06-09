import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression
df = pd.read_csv('hiring.csv')
print(df)

df.experience.fillna(0, inplace = True)

df.test_score.fillna(df.test_score.mean() , inplace = True)

X= df.iloc[:,:3]

def convert_to_int(word):
    word_dic = {'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10,'eleven':11,0:0}
    return word_dic[word]

X['experience'] = X.experience.apply(lambda x : convert_to_int(x))

y = df.iloc[:,-1]

regressor = LinearRegression()
regressor.fit(X,y)
print('Model Training is done')

joblib.dump(regressor , 'hiring._model.pkl',)
print(regressor.predict([[1,8,9]]))
