# Machine Learning Basics Course

Welcome to the **Machine Learning Basics Course**! This course is designed to provide you with a solid foundation in key machine learning algorithms and techniques. Each topic in the course is covered in a dedicated module, with explanations, code examples, and exercises to reinforce your understanding.

## Table of Contents

1. [Introduction](#introduction)
2. [Course Structure](#course-structure)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Modules](#modules)
6. [Neural Networks](#neural-networks)
7. [Data](#data)

## Introduction

This course is designed for beginners who are new to machine learning. It will walk you through the fundamental concepts and algorithms that are widely used in the industry. By the end of this course, you should be able to understand, implement, and apply these algorithms to real-world problems.

The modules cover supervised and unsupervised learning. Neural networks are covered separately.

## Course Structure

Each module has its own folder with:

- `<topic>_concept.ipynb`: the **concepts** – definitions, formulas, worked examples (computed in Python) and
  additional explanations, illustrated with small simulated examples. One concept notebook per module; it applies to
  all exercises of the module.
- `<name>_exercise.ipynb`: the **exercises** – each starts with calculations by hand (using a notebook
  cell as calculator) and continues in Python with real data. Each exercise notebook works with exactly one data set.
- `<name>_solution.ipynb`: the same notebook with complete code and written solutions.

Read the concept notebook first, then work on the exercise notebook and use the solution only to check your work. All
data sets are stored in the common folder [00_Data](00_Data/).

## Requirements

To follow along with this course, you will need a GitHub account and knowledge about how to work with GitHub Codespaces and Visual Studio Code (IDE).

## Installation

**Fork the following repository into your GitHub account:**

https://github.com/mario-gellrich-zhaw/python_machine_learning_basics

Create a new GitHub Codespaces environment based on the fork.

## Modules

Each exercise notebook works with exactly **one** data set.

| Folder | Data | Methods and concepts |
|---|---|---|
| [01_Model_Training_Evaluation](01_Model_Training_Evaluation/) | Bike counts Zurich | MSE/RMSE/MAE, train/test split, k-fold CV, under- and overfitting, bias–variance (polynomial regression), more covariates vs. more complexity |
| [02_Linear_Regression](02_Linear_Regression/) | Apartments | OLS by hand, rent vs. living area, prediction for 95 m², confidence vs. prediction interval, rooms + area (ceteris paribus, VIF), residual diagnostics, interaction with the location, more covariates and multicollinearity, maximum likelihood |
| [03_Regression_Trees_Ensembles](03_Regression_Trees_Ensembles/) | Bike counts Zurich (Mythenquai) | SSE split by hand, regression tree, pruning (`ccp_alpha`), bagging, random forest (OOB, `max_features`, permutation importance), gradient boosting, model comparison, additional covariates |
| [04_Logistic_Regression_Evaluation](04_Logistic_Regression_Evaluation/) | Avalanche accidents Switzerland | Sigmoid by hand, logistic regression, confusion matrix and metrics, cost-based threshold, ROC/AUC, information available at prediction time |
| [05_Classification_Trees_Ensembles](05_Classification_Trees_Ensembles/) | Titanic | Gini by hand, the example tree, leaf probabilities, pruning, averaging in a random forest, «Putting it together in scikit-learn», gradient boosting, ROC comparison |
| [06_kNN_Naive_Bayes_SVM](06_kNN_Naive_Bayes_SVM/) | Titanic | k-NN (scaling, choice of k), SVM (margin, support vectors, C, kernel), Naive Bayes by hand and with categorical features |
| [07_Clustering_PCA](07_Clustering_PCA/) | Bike counts Zurich | k-means by hand, features for clustering, elbow and silhouette, cluster profiles, stability, dendrogram (Ward), PCA (explained variance, loadings) |

Each folder contains `<topic>_concept.ipynb`, `<topic>_exercise.ipynb` and `<topic>_solution.ipynb`.

**Further reading:** Li, H. (2024). *Machine Learning Methods*. Springer. – scikit-learn user guide: https://scikit-learn.org/stable/user_guide.html

## Neural Networks

The folder [08_Neural_Networks](08_Neural_Networks/) contains notebooks on simple neural networks, multi-layer perceptrons and convolutional neural networks.

- **Introduction:** Overview of neural networks and their biological inspiration.
- **Mathematical Foundations:** Understanding perceptrons, activation functions, backpropagation, and gradient descent.
- **Implementation:** Building simple neural networks from scratch and using frameworks like TensorFlow or PyTorch.
- **Practical Applications:** Applying neural networks to classification and regression tasks.
- **Exercises:** Exercises to build, train, and evaluate neural networks.

- **Mathematical Foundations:** Li, H. (2023). Machine Learning Methods. Springer Nature.
- **In scikit-learn:** https://scikit-learn.org/stable/modules/neural_networks_supervised.html

## Data

All data sets are stored in the folder [00_Data](00_Data/), so no Kaggle account or download is needed. The script
[00_Data/prepare_data.py](00_Data/prepare_data.py) documents how the files were created from the original sources and can be
used to re-create them (`python 00_Data/prepare_data.py`).

| File | Content | Source |
|---|---|---|
| `apartments_data_enriched_cleaned.csv` | 774 rental apartments in the canton of Zurich (rooms, living area, rent, municipality data) | ZHAW course data (web-scraped rental listings) |
| `titanic.csv` | 891 passengers of the Titanic | [Kaggle](https://www.kaggle.com/datasets/yasserh/titanic-dataset) |
| `zurich_bike_counts_2024_2025.csv` | daily bike counts at 16 automatic counting stations in the city of Zurich (674 days in 2024–2025), with weather (station Stampfenbachstrasse), public holidays and school holidays | [Open Data Stadt Zürich](https://data.stadt-zuerich.ch/dataset/ted_taz_verkehrszaehlungen_werte_fussgaenger_velo): bike counts (Tiefbauamt), [daily weather data](https://data.stadt-zuerich.ch/dataset/ugz_meteodaten_tagesmittelwerte) (UGZ); licence CC0 |
| `zurich_bike_stations.csv` | names and directions of the 16 counting stations | [Geoportal Stadt Zürich](https://data.stadt-zuerich.ch/dataset/geo_standorte_der_automatischen_fuss__und_velozaehlungen); licence CC0 |
| `avalanche_accidents_switzerland.csv` | 4'188 avalanche accidents in Switzerland since 1970/71 (location, terrain, danger level, activity, persons caught, buried and killed) | WSL Institute for Snow and Avalanche Research SLF, [EnviDat](https://www.envidat.ch/dataset/avalanche-accidents-in-switzerland-since-1970-71); [SLF terms of use](https://www.slf.ch/en/services-and-products/data-and-monitoring/slf-data-service.html) |
| `shark_attacks_2000_2025.csv` | 2'566 shark incidents worldwide 2000–2025 (activity, species, injury, fatal) | [Global Shark Attack File](https://www.sharkattackfile.net/incidentlog.htm) (also the source of [sharkdatalab.com](https://www.sharkdatalab.com/en)) |

The shark data are currently not used in the exercises; they are kept as an additional data set (e.g. for text
classification with Naive Bayes). Note: they contain short descriptions of injuries, some of which are graphic.
