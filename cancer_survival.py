import numpy as np
import pandas as pd
import joblib
df=pd.read_csv('./cancer/kaggle_to_students.csv')
y=df[['eventdeath']]
x=df.drop(columns=['eventdeath','Patient'])
x=x.values
y=y.values
y=y.reshape(-1)
from sklearn.ensemble import RandomForestClassifier
xt=pd.read_csv('./cancer/kaggle_prediction_features.csv')
xt=xt.drop(columns=['Patient'])
model=RandomForestClassifier()
model.fit(x,y)
yt=model.predict(xt)
joblib.dump(model,'cancer_survival.pkl')