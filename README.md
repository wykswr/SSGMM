# A semi supervised approach for scRNA-seq cell type annotation based on GMM

## Background
  Single-cell RNA sequencing (scRNA-seq) can be a valuable tool for annotating cell types, but typical methods rely on marker gene databases and can be
  difficult to use with in-house data. In this research, I propose a modification to the Expectation Maximization (EM) algorithm of a Gaussian Mixture
  Model (GMM) approach, and utilize semi-supervised clustering to transfer cell type labels from reference data. This new model achieves impressive performance
  on the PBMC3k dataset, and can improve annotation quality by setting a probability threshold. It has the potential to be used to generate new cell type
  references in a high-throughput manner, making it useful for improving single-cell databases and building machine learning based annotation tools.
## Data
The PBMC3k dataset used in this study was obtained from 10x Genomics through AllCell, and consists of 2711 cells and 8664 genes from a healthy female
donor aged 25. The dataset was manually annotated using the Scanpy package, identifying 8 cell types, as shown in Figure 2. And 95\% of the cell labels
were removed to create the test set for evaluation.
## Usage
Configure the evironment:

`pip install -r requirement.txt`

Run the following in order:
1. load_data.py
2. train.py
3. visualization.py