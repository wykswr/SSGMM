import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sn
from sklearn.metrics import confusion_matrix, f1_score

from load_data import anndata

data = pd.read_csv('data/predict.csv', index_col=0)
color_map = {'CD4 T cells': 'red', 'CD14+ Monocytes': 'blue', 'B cells': 'yellow',
             'CD8 T cells': 'pink', 'NK cells': 'green', 'FCGR3A+ Monocytes': 'brown',
             'Dendritic cells': 'cyan', 'Megakaryocytes': 'purple'}


def draw_umap(data: pd.DataFrame, cutoff=0.):
    data = data.copy()
    color_map['unknown'] = 'grey'
    if not cutoff:
        sc.pl.umap(anndata, color='louvain', palette=color_map, save='origin.pdf')
        anndata.obs['semi'] = data['pre']
        sc.pl.umap(anndata, color='semi', palette=color_map, save='annotated.pdf')
        data.loc[data.test, 'pre'] = 'unknown'
        anndata.obs['before'] = data['pre']
        sc.pl.umap(anndata, color='before', palette=color_map, save='raw.pdf')
    else:
        data.loc[data.prob < cutoff, 'pre'] = 'unknown'
        anndata.obs['semi'] = data['pre']
        sc.pl.umap(anndata, color='semi', palette=color_map, save='annotated_cutoff_{}.pdf'.format(cutoff))


def heatmap(data):
    ct = [x for x in color_map.keys() if x != 'unknown']
    cm = confusion_matrix(data['cell'], data['pre'], labels=ct)
    df_cm = pd.DataFrame(cm, index=ct, columns=ct)
    sn.heatmap(df_cm, annot=True)
    plt.subplots_adjust(bottom=0.35, left=.25)
    plt.savefig('figures/cm.pdf')
    plt.clf()


def f1_change(data, cutoffs) -> list:
    res = list()
    for cut in cutoffs:
        sub = data.loc[data.prob > cut]
        res.append(f1_score(sub['cell'], sub['pre'], average='macro'))
    return res


if __name__ == '__main__':
    draw_umap(data)
    plt.style.use('ggplot')
    plt.hist(data['prob'], bins=150)
    plt.xlabel('probability')
    plt.ylabel('count')
    plt.savefig('figures/hist.pdf')
    plt.clf()

    thresholds = np.linspace(0, .95, 50)
    plt.style.use('ggplot')
    plt.plot(thresholds, f1_change(data, thresholds), color='blue')
    plt.xlabel('probability cutoff')
    plt.ylabel('macro F1')
    plt.savefig('figures/f1.pdf')
    plt.clf()
    heatmap(data)