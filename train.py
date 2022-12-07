import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from SSGMM import SSGaussianMixture


data = pd.read_csv('data/umap.csv', index_col=0)
le = joblib.load("data/encoder.joblib")
train, test = train_test_split(data, test_size=.95, random_state=404)
labels = np.array(train.iloc[:, -1])
n_cat = len(train.iloc[:, -1].unique())
assert n_cat == 8
n_dim = 2
ssgmm = SSGaussianMixture(n_cat)
ssgmm.fit(train.iloc[:, :n_dim].values, labels, test.iloc[:, :n_dim].values)

test['pre'] = ssgmm.predict(test.iloc[:, :n_dim].values)
probs = ssgmm.predict_proba(test.iloc[:, :n_dim].values)
pre_labels = le.inverse_transform(pd.concat([train['cell'], test['pre']])[data.index])
pre_labels = pd.Series(pre_labels, index=data.index)
data['cell'] = le.inverse_transform(data['cell'])
data['pre'] = pre_labels
data['test'] = False
data.loc[test.index, 'test'] = True
data['prob'] = 1
data.loc[test.index, 'prob'] = probs
data.to_csv('data/predict.csv', index=True)
