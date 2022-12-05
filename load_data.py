import scanpy
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib


anndata = scanpy.datasets.pbmc3k_processed()
labels = anndata.obs['louvain']
le = LabelEncoder()
represents = pd.DataFrame(anndata.obsm['X_pca'], index=labels.index)
represents['cell'] = le.fit_transform(labels)
represents.to_csv('pbmc_3k.csv', index=True)
joblib.dump(le, 'data/encoder.joblib')
