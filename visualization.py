from load_data import anndata
import matplotlib.pyplot as plt
import scanpy as sc


sc.pl.umap(anndata, color='louvain')
plt.show()