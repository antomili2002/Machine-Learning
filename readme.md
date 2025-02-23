# Machine Learning Projects

This repository contains a collection of machine learning assignments and projects I worked on. Each project demonstrates a fundamental ML or data analysis concept, using popular Python libraries (such as NumPy, matplotlib, scikit-learn, etc.). The datasets range from synthetic to more complex datasets like MNIST and 20 Newsgroups.

Below is an overview of each project, including key features and what you can learn or see demonstrated in the code.

---

## Table of Contents

- [Machine Learning Projects](#machine-learning-projects)
  - [Table of Contents](#table-of-contents)
  - [Naive Bayes on MNIST](#naive-bayes-on-mnist)
  - [Softmax Regression (Multiclass) on MNIST](#softmax-regression-multiclass-on-mnist)
  - [Gaussian Mixture Models Clustering on Synthetic Data](#gaussian-mixture-models-clustering-on-synthetic-data)
  - [Kernelized Logistic Regression on Two Moons Dataset](#kernelized-logistic-regression-on-two-moons-dataset)
  - [Pseudo Inverse for Minimizing ||XW - Y||\_2](#pseudo-inverse-for-minimizing-xw---y_2)
  - [PCA using SVD on MNIST (Visualization)](#pca-using-svd-on-mnist-visualization)
  - [Latent Semantic Analysis on 20 Newsgroups Dataset](#latent-semantic-analysis-on-20-newsgroups-dataset)
  - [L1-Regularized Kernelized Logistic Regression](#l1-regularized-kernelized-logistic-regression)
  - [Kernelized Ridge Regression](#kernelized-ridge-regression)
  - [Spectral Clustering](#spectral-clustering)
  - [Contact](#contact)
    - [Thanks for Visiting](#thanks-for-visiting)

---

## Naive Bayes on MNIST

- **Goal:** Classify digits (0–9) from the MNIST dataset using a Naive Bayes approach.
- **Key Steps:**
  - Data loading and preprocessing (flattening images, normalizing pixel values).
  - Implementing Gaussian or Multinomial Naive Bayes from scratch.
  - Evaluating accuracy.
  - Generating new data and handwritten digits.
- **Highlights:**
  - Shows how a simple probabilistic classifier can perform well on digit classification.

---

## Softmax Regression (Multiclass) on MNIST

- **Goal:** Extend logistic regression to multiple classes using a softmax layer.
- **Key Steps:**
  - Manual implementation to handle multiple classes.
  - Gradient descent, stochastic gradient descent and newtons optimizer.
  - Accuracy and loss tracking through epochs.
- **Highlights:**
  - Illustrates fundamental concepts of multinomial logistic regression.
  - Insight into convergence plots and confusion matrices.

---

## Gaussian Mixture Models Clustering on Synthetic Data

- **Goal:** Cluster a synthetic dataset (e.g., generated via `make_blobs`).
- **Key Steps:**
  - Implement GaussianMixtureModel.
  - Visualize the decision boundaries and clusters in 2D.
  - Visualize each Iteration to show the working of the estimation of the parameters
- **Highlights:**
  - Introduction to probabilistic clustering.
  - Demonstrates how to interpret mixture component parameters.

---

## Kernelized Logistic Regression on Two Moons Dataset

- **Goal:** Classify the two moons dataset using a nonlinear decision boundary.
- **Key Steps:**
  - Utilize a Kernel-Machine using random centroids to map data to a higher-dimensional feature space.
  - Plot decision boundary and measure performance against normal logistic regression.
- **Highlights:**
  - Shows how kernels can transform a linearly inseparable problem into a separable one.
  - Emphasizes hyperparameter tuning for kernel parameters.

---

## Pseudo Inverse for Minimizing \|\|XW - Y\|\|_2

- **Goal:** Solve linear regression via the Moore-Penrose pseudo-inverse directly.
- **Key Steps:**
  - Use `numpy.linalg.pinv(X)` to compute `W = pinv(X) * Y`.
- **Highlights:**
  - Illustrates an analytical approach to solving linear regression.
  - Educational perspective on linear algebra for ML.

---

## PCA using SVD on MNIST (Visualization)

- **Goal:** Reduce high-dimensional MNIST data for visualization in 2D/3D.
- **Key Steps:**
  - Compute covariance matrix or directly use SVD (`numpy.linalg.svd`).
  - Project the data onto principal components.
  - Plot the transformed data in 2D color-coded by digit label.
- **Highlights:**
  - Demonstrates how PCA can reveal latent structure in high-dimensional data.
  - Useful for dimensionality reduction and data compression insights.

---

## Latent Semantic Analysis on 20 Newsgroups Dataset

- **Goal:** Perform LSA (a variant of Truncated SVD) to uncover topics in text data.
- **Key Steps:**
  - Vectorize documents (count or TF-IDF).
  - Apply SVD on the document-term matrix.
  - Interpret the resulting latent topics.
- **Highlights:**
  - Shows how LSA can capture semantic relationships between words and documents.
  - Foundational concept for modern NLP methods (e.g., topic modeling).

---

## L1-Regularized Kernelized Logistic Regression

- **Goal:** Extend the kernelized logistic regression with L1 penalty to encourage sparsity and feature selection.
- **Key Steps:**
  - Incorporate L1 penalty (`|w|`) in the objective function.
  - Visualize Loss over Iterations and selecting hyperparameters.
- **Highlights:**
  - Demonstrates how L1 regularization encourages sparse solutions.

---

## Kernelized Ridge Regression

- **Goal:** Fit a linear model with an L2 penalty on the coefficients.
- **Key Steps:**
  - Implement using numpy and projecting the data using the kernel trick into a higher dimnsional space to retrieve non-linear relations.
  - Select and observe accurancy of polynomial and rbf_kernel.
  - Tune the regularization parameters.
  - Evaluation on housing dataset
- **Highlights:**
  - Illustrates the effect of L2 regularization on coefficient shrinkage.
  - Explains improved numerical stability and reduced overfitting.
  - Makes usage of kernel trick for dot-products in higher dimensional space using mercer-kernel.

---

## Spectral Clustering

- **Goal:** Cluster data by using the graph Laplacian and its eigenvectors.
- **Key Steps:**
  - Construct similarity matrix from data points.
  - Compute the Laplacian and its smallest eigenvalues/eigenvectors.
  - Apply K-Means on the eigenvector embedding to get final clusters.
- **Highlights:**
  - Demonstrates advanced clustering when data is not well-served by centroid-based methods.
  - Useful for complex cluster shapes and manifold-based data.

---

## Contact

If you have questions or feedback, feel free to open an issue or pull request. I’m constantly learning and welcome any collaboration suggestions.

---

### Thanks for Visiting

I hope these projects illustrate my understanding of various machine learning techniques, from classic parametric methods to more modern regularization and dimensionality reduction approaches.
