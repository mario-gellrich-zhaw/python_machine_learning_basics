# Machine Learning Basics Course

Welcome to the **Machine Learning Basics Course**! This course is designed to provide you with a solid foundation in key machine learning algorithms and techniques. Each topic in the course is covered in a dedicated module, with explanations, code examples, and exercises to reinforce your understanding.

## Table of Contents

1. [Introduction](#introduction)
2. [Course Structure](#course-structure)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Machine Learning Algorithms](#machine-learning-algorithms)
   - [Linear Regression](#linear-regression)
   - [Logistic Regression](#logistic-regression)
   - [Classification and Regression Trees (CART)](#cart)
   - [Random Forest](#random-forest)
   - [K-Nearest Neighbors (KNN)](#knn)
   - [K-Means Clustering](#k-means)
   - [Neural Networks](#neural-networks)
6. [Exercises Week 06](#exercises-week-06)

## Introduction

This course is designed for beginners who are new to machine learning. It will walk you through the fundamental concepts and algorithms that are widely used in the industry. By the end of this course, you should be able to understand, implement, and apply these algorithms to real-world problems.

## Course Structure

Each explanation of Machine Learning (ML) algorithms follows a consistent structure:

- **Introduction:** A brief overview of the algorithm, including its purpose and use cases.
- **Mathematical Foundations:** The key mathematical concepts and equations behind the algorithm.
- **Implementation:** Step-by-step code examples to implement the algorithm from scratch.
- **Practical Applications:** How to apply the algorithm to real-world datasets.
- **Exercises:** Hands-on exercises to reinforce the concepts learned.

## Requirements

To follow along with this course, you will need a GitHub account and knowledge about how to work with GitHub Codespaces and Visual Studio Code (IDE).

## Installation

**Fork the following repository into your GitHub account:**

https://github.com/mario-gellrich-zhaw/python_machine_learning_basics

Create a new GitHub Codespaces environment based on the fork.

## Machine Learning Algorithms

### Linear Regression

Linear Regression is one of the simplest and most commonly used algorithms in machine learning. It is used to model the relationship between a dependent variable and one or more independent variables.

- **Introduction:** Understanding the concept of linear relationships and the line of best fit.
- **Mathematical Foundations:** Deriving the linear regression equation, loss function, and optimization techniques.
- **Implementation:** Code examples using Python to implement linear regression.
- **Practical Applications:** Applying linear regression to predict housing prices, sales forecasting, etc.
- **Exercises:** Practice problems to solidify your understanding.

- **Mathematical Foundations:** Martin, P. (2022). Linear regression: An introduction to statistical models.
- **In scikit-learn:** https://scikit-learn.org/stable/modules/linear_model.html

### Logistic Regression

Logistic Regression is a method used for classification. In a binary case it is used to predict the probability of a binary outcome based on one or more predictor variables.

- **Introduction:** Understanding the difference between linear and logistic regression.
- **Mathematical Foundations:** Deriving the sigmoid function and the cost function for logistic regression.
- **Implementation:** Python code examples to implement logistic regression from scratch.
- **Practical Applications:** Applying logistic regression to classification tasks.
- **Exercises:** Hands-on exercises to practice logistic regression.

- **Mathematical Foundations:** Li, H. (2023). Machine Learning Methods. Springer Nature.
- **In scikit-learn:** https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression

### Classification and Regression Trees (CART)

CART is a powerful algorithm used for both classification and regression tasks. It creates a tree-like model of decisions based on input features.

- **Introduction:** Understanding decision trees and their applications.
- **Mathematical Foundations:** Gini impurity (CT), Variance reduction (RT) and tree pruning techniques.
- **Implementation:** Building a decision tree from scratch using Python.
- **Practical Applications:** Using CART for classification and regression tasks.
- **Exercises:** Exercises to build and evaluate decision trees.

- **Mathematical Foundations:** Li, H. (2023). Machine Learning Methods. Springer Nature.
- **In scikit-learn:** https://scikit-learn.org/stable/modules/tree.html

### Random Forest

Random Forest is an ensemble learning method that combines multiple decision trees to improve accuracy and prevent overfitting.

- **Introduction:** Overview of ensemble learning and the concept of bagging.
- **Mathematical Foundations:** Understanding the random forest algorithm and its parameters.
- **Implementation:** Implementing random forests using Python's `scikit-learn` library.
- **Practical Applications:** Applying random forests to classification and regression tasks.
- **Exercises:** Practice problems to explore the power of random forests.

- **Mathematical Foundations:** Li, H. (2023). Machine Learning Methods. Springer Nature.
- **In scikit-learn:** https://scikit-learn.org/stable/modules/ensemble.html

### K-Nearest Neighbors (KNN)

KNN is a simple, non-parametric algorithm used for both classification and regression tasks. It makes predictions based on the closest training examples in the feature space.

- **Introduction:** Overview of the KNN algorithm and its use cases.
- **Mathematical Foundations:** Understanding distance metrics and decision boundaries.
- **Implementation:** Coding KNN from scratch and using libraries like `scikit-learn`.
- **Practical Applications:** Applying KNN to classification and regression tasks.
- **Exercises:** Exercises to practice and fine-tune KNN models.

- **Mathematical Foundations:** Li, H. (2023). Machine Learning Methods. Springer Nature.
- **In scikit-learn:** https://scikit-learn.org/stable/modules/neighbors.html

### K-Means Clustering

K-Means is an unsupervised learning algorithm used to partition a dataset into distinct clusters based on feature similarity.

- **Introduction:** Understanding clustering and the K-Means algorithm.
- **Mathematical Foundations:** The K-Means objective function and the algorithm's iterative process.
- **Implementation:** Implementing K-Means clustering from scratch and using Python libraries.
- **Practical Applications:** Applying K-Means to clustering tasks.
- **Exercises:** Practice problems to explore different clustering scenarios.

- **Mathematical Foundations:** Li, H. (2023). Machine Learning Methods. Springer Nature.
- **In scikit-learn:** https://scikit-learn.org/stable/modules/clustering.html#k-means

### Neural Networks.
- **Introduction:** Overview of neural networks and their biological inspiration.
- **Mathematical Foundations:** Understanding perceptrons, activation functions, backpropagation, and gradient descent.
- **Implementation:** Building simple neural networks from scratch and using frameworks like TensorFlow or PyTorch.
- **Practical Applications:** Applying neural networks classification and regression tasks.
- **Exercises:** Exercises to build, train, and evaluate neural networks.

- **Mathematical Foundations:** Li, H. (2023). Machine Learning Methods. Springer Nature.
- **In scikit-learn:** https://scikit-learn.org/stable/modules/neural_networks_supervised.html

## Exercises Week 06

The folder [Exercises_Week_06](Exercises_Week_06/) contains seven exercises with real data sets for the slides
*Week 06: Supervised Learning*. They cover learning from data, linear regression, regression trees and ensembles, logistic
regression and evaluation, classification trees and ensembles, k-NN, Naive Bayes, SVM, clustering and PCA. Each
exercise has a short concept recap, an exercise notebook and a solution notebook. See the
[overview](Exercises_Week_06/README.md) for details and data sources.
