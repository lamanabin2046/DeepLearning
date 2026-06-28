# Assignment 5: Graph Neural Networks

## Graph Neural Networks (GCN, GAT, GraphSAGE)

This repository contains my implementation and experimental analysis of several **Graph Neural Network (GNN)** models using **PyTorch**. The assignment investigates over-smoothing in Graph Convolutional Networks (GCNs), compares different GNN architectures, and evaluates the importance of graph structure through an MLP baseline.

---

# Objectives

* Implement a variable-depth Graph Convolutional Network (GCN)
* Investigate the over-smoothing phenomenon
* Compare GCN, GAT, and GraphSAGE
* Visualize Graph Attention Network (GAT) attention weights
* Compare graph-based models with a standard Multi-Layer Perceptron (MLP)

---

# Repository Structure

```text
A5/
├── A5-Graph-Neural-Networks.ipynb
├── oversmoothing.png
└── tsne_comparison.png
```

---

# Models Implemented

* Graph Convolutional Network (GCN)
* Graph Attention Network (GAT)
* GraphSAGE
* Multi-Layer Perceptron (MLP)

---

# Experimental Results

| Model          | Test Accuracy |
| -------------- | ------------: |
| MLP (No Graph) |    **96.80%** |
| GraphSAGE      |    **67.00%** |
| GCN            |    **30.00%** |
| GAT            |     **1.20%** |

---

# Over-smoothing Analysis

As the number of GCN layers increases, node embeddings become increasingly similar (higher cosine similarity), illustrating the over-smoothing phenomenon. This generally reduces the model's ability to distinguish between node classes.

![Over-smoothing](A5/oversmoothing.png)

---

# Embedding Visualization

The learned node embeddings are visualized using t-SNE.

![t-SNE Visualization](A5/tsne_comparison.png)

---

# Technologies

* Python
* PyTorch
* NumPy
* Matplotlib
* Jupyter Notebook

---

# Dataset

MovieLens 100K

---

# How to Run

```bash
git clone <repository-url>
cd Assignment/A5

jupyter notebook A5-Graph-Neural-Networks.ipynb
```

Run the notebook cells sequentially.

---

# Author

**Nabin Lama**

Master of Science in Data Science and Artificial Intelligence
