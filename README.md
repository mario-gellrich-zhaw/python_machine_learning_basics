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

Modules 01–07 each have three notebooks:

- `<topic>_concept.ipynb`: the idea in plain language, the most important formulas, a small worked example and provided demonstration code.
- `<topic>_exercise.ipynb`: guided practice with one data set. Data preparation and plots are provided; fill a few marked code gaps and interpret the results.
- `<topic>_solution.ipynb`: the same steps with complete code and short explanations.

Read the concept notebook first. Run the exercise cells from top to bottom, replacing each `None` marked with a `# TODO` comment with the expression requested above its cell. These gaps are intentional; the affected cells cannot be used until completed. Write one or two sentences for each interpretation question, then use the solution to check your work.

Basic Python knowledge is enough: variables, arithmetic, selecting data columns and calling a function. Loops for plots, cross-validation and small parameter comparisons are supplied. Mathematical symbols are explained beside the formulas; lengthy derivations are not required.

All data sets are stored in [00_Data](00_Data/). The notebooks assume that their working directory is their own module folder, so the data paths start with `../00_Data/`.

## Requirements

To follow along with this course, you will need a GitHub account and knowledge about how to work with GitHub Codespaces and Visual Studio Code (IDE).

## Installation

**Fork the following repository into your GitHub account:**

https://github.com/mario-gellrich-zhaw/python_machine_learning_basics

Create a new GitHub Codespaces environment based on the fork. The Codespace uses Python 3.11 and installs all required
packages automatically (from `requirements.txt`) when it is created.

**Working on your own computer instead:** install Python 3.11 and then the required packages with

```bash
pip install -r requirements.txt
```

## Modules

Each exercise notebook works with one data set. The modules follow the topics and learning objectives of [the lecture slides](Slides_MSc_Data_Science_Week_06_EN.pdf). The slide references appear in the concept notebooks.

| Folder | Data | Core topics | Slides |
|---|---|---|---|
| [01_Model_Training_Evaluation](01_Model_Training_Evaluation/) | Car fuel consumption 2025 | ML paradigms and history; loss, objective and metric; MSE/RMSE/MAE; train/test; 5-fold CV; under-/overfitting and bias–variance | 5–17 |
| [02_Linear_Regression](02_Linear_Regression/) | Zurich rental apartments | OLS; coefficients; simple and multiple regression; R²/RMSE; confidence and prediction intervals; assumptions and residual diagnostics; maximum likelihood and gradient descent | 19–27 |
| [03_Regression_Trees_Ensembles](03_Regression_Trees_Ensembles/) | Car fuel consumption 2025 | SSE splits and leaf means; tree parameters and pruning; bagging and OOB; random forest and permutation importance; boosting; CV comparison | 28–37 |
| [04_Logistic_Regression_Evaluation](04_Logistic_Regression_Evaluation/) | Swiss avalanche accident records | Linear score and sigmoid; log-loss; confusion matrix; accuracy, precision, recall, F1 and FPR; fixed thresholds and error costs; ROC/AUC; input availability | 39, 48–50 |
| [05_Classification_Trees_Ensembles](05_Classification_Trees_Ensembles/) | Titanic | Gini split; tree rules and leaf probabilities; pruning; random forest and boosting; fully provided small GridSearchCV example | 40–44, 57 |
| [06_kNN_Naive_Bayes_SVM](06_kNN_Naive_Bayes_SVM/) | Titanic | Neighbours and distance; scaling; a small k comparison; Bayes calculation and categorical Naive Bayes; SVM margin, support vectors, C and kernels | 45–47 |
| [07_Clustering_PCA](07_Clustering_PCA/) | Car fuel consumption 2025 | Unsupervised learning; a k-means update; scaled car features; elbow plot and cluster interpretation; dendrogram; PCA outlook and explained variance | 52–56 |

The worked calculations, guided exercises and interpretation questions cover the learning objectives on slide 58. Topics such as cross-validation, grid search and plotting have complete example code so that students can focus on the model and its results.

The avalanche notebook models fatal outcomes **among recorded accidents** with burial information available after an accident. It does not predict avalanche occurrence before a trip. Titanic examples use rows with known age and fare for simple preparation; excluding missing rows may affect representativeness.

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
used to re-create them (`python 00_Data/prepare_data.py`). The CSV files in the repository are the reference versions
used in the notebooks; the original sources may change or become unavailable over time.

| File | Content | Source |
|---|---|---|
| `rental_apartments_canton_zh.csv` | 774 rental apartments in the canton of Zurich (rooms, living area, rent, municipality data) | ZHAW course data (web-scraped rental listings) |
| `titanic.csv` | 891 passengers of the Titanic | [Kaggle](https://www.kaggle.com/datasets/yasserh/titanic-dataset) |
| `car_fuel_consumption_2025.csv` | 701 new car models of model year 2025 (make, model, vehicle class, engine size, cylinders, transmission, fuel type, fuel consumption city/highway/combined in L/100 km, km per litre, CO₂ emissions) | Natural Resources Canada, [Fuel consumption ratings](https://open.canada.ca/data/en/dataset/98f1a129-f628-4ce4-b24d-6f16bf24dd64); [Open Government Licence – Canada](https://open.canada.ca/en/open-government-licence-canada) |
| `avalanche_accidents_switzerland.csv` | 4'188 avalanche accidents in Switzerland since 1970/71 (location, terrain, danger level, activity, persons caught, buried and killed) | WSL Institute for Snow and Avalanche Research SLF, [EnviDat](https://www.envidat.ch/dataset/avalanche-accidents-in-switzerland-since-1970-71); [SLF terms of use](https://www.slf.ch/en/services-and-products/data-and-monitoring/slf-data-service.html) |
