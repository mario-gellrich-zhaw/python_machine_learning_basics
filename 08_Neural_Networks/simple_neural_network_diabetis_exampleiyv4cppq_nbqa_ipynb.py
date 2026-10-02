# %%NBQA-CELL-SEP42d9ce
# Libraries
import os
import platform
import warnings
from datetime import datetime
from platform import python_version

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prettytable import PrettyTable

# Ignore warnings
warnings.filterwarnings('ignore')

# Show current working directory
print(os.getcwd())


# %%NBQA-CELL-SEP42d9ce
# Function to generate pretty tables
def generate_ascii_table(df):
    """Print a data frame as an ASCII table and return the table."""
    x = PrettyTable()
    x.field_names = df.columns.tolist()
    for row in df.values:
        x.add_row(row)
    print(x)
    return x


# Activation function (sigmoid)
def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))


# Derivative of the activation function
# -> When updating the curve, to know in which
#    direction and how much to change or update


#    the curve depending upon the slope.
def sigmoid_der(x):
    """Derivative of the sigmoid function."""
    return sigmoid(x) * (1 - sigmoid(x))


# %%NBQA-CELL-SEP42d9ce
# Create data frame
data = np.array(
    [[0, 1, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0], [1, 1, 0, 1], [1, 1, 1, 1]]
)
df = pd.DataFrame(data, columns=['Smoking', 'Obesity', 'Exercise', 'Diabetic'])

# Show table
generate_ascii_table(df)

# Matrix with features (X) and labels (y)
X = data[0:5, 0:3]
y = data[0:5, 3]
y = y.reshape(5, 1)


# %%NBQA-CELL-SEP42d9ce
# Plots the sigmoid function and its derivative
z = np.linspace(-10, 10, 100)
plt.plot(z, sigmoid(z), color="green")
plt.plot(z, sigmoid(z) * (1 - sigmoid(z)), color="orange")
plt.grid(color='gray', linestyle='-', linewidth=0.1)


# %%NBQA-CELL-SEP42d9ce
# Initialize weights and bias & define learning rate und the number of epochs
np.random.seed(42)
weights = np.random.rand(3, 1)
bias = np.random.rand(1)
learning_rate = 0.25
num_epochs = 200


# %%NBQA-CELL-SEP42d9ce
# Calculations step by step

# Inputs
inputs = X

# Print X, y, weights and bias
print('X:', '\n', inputs, '\n')
print('y:', '\n', y, '\n')
print('weights:', '\n', weights, '\n')
print('bias:', '\n', bias, '\n')

# Calculate X * weigths + bias
XW = np.dot(X, weights) + bias
print('XW:', '\n', XW, '\n')

# Apply activation function to XW
z = sigmoid(XW)
print('sigmoid(XW) = z', '\n', z, '\n')

# Calculate the error (difference between labels and z)
error = z - y
print('error:', '\n', error, '\n')

# Backpropagation (weights and bias adaptation)
dcost_dpred = error
dpred_dz = sigmoid_der(z)
print('sigmoid_der(z):', '\n', dpred_dz, '\n')

# Calculate the delta of z (chain rule)
z_delta = dcost_dpred * dpred_dz
print('z_delta:', '\n', z_delta, '\n')

# Adapt weights
inputs = X.T
weights -= learning_rate * np.dot(inputs, z_delta)
print('adapted weights', '\n', weights, '\n')

# Adapt bias
print('adapted bias:')
for num in z_delta:
    bias -= learning_rate * num
    print(bias)


# %%NBQA-CELL-SEP42d9ce
# Neural network

# Prepare figure
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('Epoche')
ax.set_ylabel('Error')
plt.title('Error per Epoche')
plt.ion()

# Show the empty figure
fig.show()
fig.canvas.draw()

# --------------------------------------
# Loop with backpropagation
# --------------------------------------

# Initialize list
d = []

# Train the network for num_epochs epochs
for epoch in range(num_epochs):

    inputs = X

    # Calculate (X * weigths) + bias
    XW = np.dot(X, weights) + bias

    # Apply activation function to XW
    z = sigmoid(XW)

    # Calculate the error (difference between label-value and z)
    error = z - y

    # Store error in data frame
    d.append({'Epoch': epoch, 'Error': error.sum()})
    df = pd.DataFrame(d)

    # Backpropagation (change weights and bias)
    dcost_dpred = error
    dpred_dz = sigmoid_der(z)
    z_delta = dcost_dpred * dpred_dz

    inputs = X.T
    weights -= learning_rate * np.dot(inputs, z_delta)

    for num in z_delta:
        bias -= learning_rate * num

    # -----------------------
    # Plot error per epoche
    # -----------------------
    plt.xlim([0, num_epochs])
    plt.ylim([0, df.Error[0]])
    ax.plot(df.Epoch, df.Error)
    fig.canvas.draw()


# %%NBQA-CELL-SEP42d9ce
# Model-prediction based on new data
data_new = np.array([[0, 1, 0]])
df_new = pd.DataFrame(data_new, columns=['Smoking', 'Obesity', 'Exercise'])

# Show the new data
print('New Data (basis for prediction):', '\n')
generate_ascii_table(df_new)

# Calculate probability of having Diabetes
result = sigmoid(np.dot(data_new, weights) + bias)
print('\n', 'Predicted probability of having Diabetes:', result)


# %%NBQA-CELL-SEP42d9ce
# Show operating system, date/time and Python version
print('-----------------------------------')
print(os.name.upper())
print(platform.system(), '|', platform.release())
print('Datetime:', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print('Python Version:', python_version())
print('-----------------------------------')
