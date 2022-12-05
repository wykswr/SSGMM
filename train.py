import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from SSGMM import SSGaussianMixture


data = pd.read_csv('data/pbmc_3k.csv', index_col=0)
le = joblib.load("data/encoder.joblib")
train, test = train_test_split(data, test_size=.9, random_state=404)
labels = np.array(train.iloc[:, -1])
n_cat = len(train.iloc[:, -1].unique())
n_dim = 2
ssgmm = SSGaussianMixture(n_cat)
ssgmm.fit(train.iloc[:, :n_dim].values, labels, test.iloc[:, :n_dim].values)
train['pre'] = train['cell']
train['pre'] = ssgmm.predict(test.iloc[:, :n_dim].values)
