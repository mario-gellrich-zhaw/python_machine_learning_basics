# Machine Learning Basics Course

Welcome to the **Machine Learning Basics Course**! This course is designed to provide you with a solid foundation in key machine learning algorithms and techniques. Each topic in the course is covered in a dedicated module, with explanations, code examples, and exercises to reinforce your understanding.

## Table of Contents

1. [Introduction](#introduction)
2. [Course Structure](#course-structure)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Modules](#modules)
6. [Neural Networks](#neural-networks)
7. [Data Sources](#data-sources)

## Introduction

This course is designed for beginners who are new to machine learning. It will walk you through the fundamental concepts and algorithms that are widely used in the industry. By the end of this course, you should be able to understand, implement, and apply these algorithms to real-world problems.

The modules accompany the slides *Week 06: Supervised Learning* (Parts 02–05). Neural networks are covered separately.

## Course Structure

Each module has its own folder:

- `*_exercise.ipynb`: **Concept** (a short recap of the slides), followed by the **Exercises**. Each exercise starts with calculations by hand, as on the slides, and continues in Python with real data.
- `*_solution.ipynb`: the same notebook with complete code and written solutions.
- `Data/`: the data used in the module.

Work on the exercise notebook first and use the solution only to check your work.

## Requirements

To follow along with this course, you will need a GitHub account and knowledge about how to work with GitHub Codespaces and Visual Studio Code (IDE).

## Installation

**Fork the following repository into your GitHub account:**

https://github.com/mario-gellrich-zhaw/python_machine_learning_basics

Create a new GitHub Codespaces environment based on the fork.

## Modules

| # | Folder | Slides | Dataset | Methods and concepts |
|---|---|---|---|---|
| 01 | [01_Learning_from_Data](01_Learning_from_Data/) | Part 02 | Capital Bikeshare daily rentals, Washington D.C. | MSE/RMSE/MAE, train/test split, k-fold CV, under- and overfitting, bias–variance (polynomial regression) |
| 02 | [02_Linear_Regression](02_Linear_Regression/) | Part 03 | Medical insurance costs (US) | OLS by hand, simple and multiple regression, R², RMSE, residual diagnostics, interaction, confidence vs. prediction interval, VIF, maximum likelihood |
| 03 | [03_Regression_Trees_Ensembles](03_Regression_Trees_Ensembles/) | Part 03 | Concrete compressive strength | SSE split by hand, regression tree, pruning (`ccp_alpha`), bagging, random forest (OOB, `max_features`, permutation importance), gradient boosting, model comparison |
| 04 | [04_Logistic_Regression_Evaluation](04_Logistic_Regression_Evaluation/) | Part 04 | Telco customer churn | Sigmoid by hand, logistic regression, confusion matrix and metrics, cost-based threshold, ROC/AUC |
| 05 | [05_Classification_Trees_Ensembles](05_Classification_Trees_Ensembles/) | Part 04 | Heart disease (5 hospitals) | Gini by hand, classification tree, pruning, random forest and gradient boosting with `GridSearchCV`, ROC comparison, threshold for target recall |
| 06 | [06_kNN_Naive_Bayes_SVM](06_kNN_Naive_Bayes_SVM/) | Part 04 | Palmer penguins; SMS Spam Collection | k-NN (scaling, choice of k), SVM (margin, support vectors, C, kernel), Naive Bayes by hand and as a spam filter |
| 07 | [07_Clustering_PCA](07_Clustering_PCA/) | Part 05 | Country indicators (HELP International) | k-means by hand, elbow and silhouette, cluster profiles, stability, dendrogram (Ward), PCA (explained variance, loadings) |

**Further reading:** Li, H. (2024). *Machine Learning Methods*. Springer. – scikit-learn user guide: https://scikit-learn.org/stable/user_guide.html

## Neural Networks

The folder [Neural_Networks](Neural_Networks/) contains notebooks on simple neural networks, multi-layer perceptrons and convolutional neural networks.

- **Introduction:** Overview of neural networks and their biological inspiration.
- **Mathematical Foundations:** Understanding perceptrons, activation functions, backpropagation, and gradient descent.
- **Implementation:** Building simple neural networks from scratch and using frameworks like TensorFlow or PyTorch.
- **Practical Applications:** Applying neural networks to classification and regression tasks.
- **Exercises:** Exercises to build, train, and evaluate neural networks.

- **Mathematical Foundations:** Li, H. (2023). Machine Learning Methods. Springer Nature.
- **In scikit-learn:** https://scikit-learn.org/stable/modules/neural_networks_supervised.html

## Data Sources

All data sets of modules 01–07 are stored in the `Data/` folders, so no Kaggle account is needed.

| Dataset | Reference / source |
|---|---|
| Bike sharing | Fanaee-T, H. & Gama, J. (2013). Event labeling combining ensemble detectors and background knowledge. *Progress in Artificial Intelligence*. [UCI](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset), [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/bike-sharing-dataset) |
| Medical insurance costs | Lantz, B. *Machine Learning with R*. Packt. [Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance) |
| Concrete compressive strength | Yeh, I.-C. (1998). Modeling of strength of high-performance concrete using artificial neural networks. *Cement and Concrete Research*, 28(12). [UCI](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength), [Kaggle](https://www.kaggle.com/datasets/maajdl/yeh-concret-data) |
| Telco customer churn | IBM sample data. [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| Heart failure prediction | fedesoriano (2021), combined from the UCI heart disease data sets. [Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) |
| Palmer penguins | Gorman, K. B., Williams, T. D. & Fraser, W. R. (2014). *PLoS ONE*, 9(3); Horst, A. M., Hill, A. P. & Gorman, K. B. (2020). palmerpenguins R package. [Kaggle](https://www.kaggle.com/datasets/parulpandey/palmer-archipelago-antarctica-penguin-data) |
| SMS Spam Collection | Almeida, T. A., Gómez Hidalgo, J. M. & Yamakami, A. (2011). Contributions to the study of SMS spam filtering. *ACM DocEng*. [UCI](https://archive.ics.uci.edu/dataset/228/sms+spam+collection), [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) |
| Country data | HELP International case study. [Kaggle](https://www.kaggle.com/datasets/rohan0301/unsupervised-learning-on-country-data) |

Note: the SMS data contain real, unfiltered text messages, some of which use colloquial or offensive language.
