# %%NBQA-CELL-SEP42d9ce
# Libraries
import os
import logging
import platform
import warnings
from datetime import datetime
from platform import python_version

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Set TensorFlow log level to suppress warnings (must be set before
# TensorFlow is imported)
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# TensorFlow and Keras (imported after setting the log level)
# pylint: disable=wrong-import-position
from tensorflow import keras

# pylint: enable=wrong-import-position

# Ignore warnings
warnings.filterwarnings('ignore')

# Show current working directory
print(os.getcwd())


# %%NBQA-CELL-SEP42d9ce
# Define columns for import
columns = [
    'web-scraper-order',
    'address_raw',
    'rooms',
    'area',
    'luxurious',
    'price',
    'price_per_m2',
    'lat',
    'lon',
    'bfs_number',
    'bfs_name',
    'pop',
    'pop_dens',
    'frg_pct',
    'emp',
    'mean_taxable_income',
    'dist_supermarket',
]

# Read and select variables
df_orig = pd.read_csv(
    "../00_Data/apartments_data_enriched_cleaned.csv", sep=";", encoding='utf-8'
)[columns]

# Rename variable 'web-scraper-order' to 'apmt_id'
df_orig = df_orig.rename(columns={'web-scraper-order': 'id'})

# Remove missing values
df = df_orig.dropna()
df.head(5)

# Remove duplicates
df = df.drop_duplicates()

# Remove some 'extreme' values
df = df.loc[(df['price'] >= 1000) & (df['price'] <= 5000)]

# Reset index
df = df.reset_index(drop=True)

# Shape and first rows of the cleaned data
print(df.shape)
df.head(5)


# %%NBQA-CELL-SEP42d9ce
# List of features to re-scale
features_to_scale = [
    'area',
    'rooms',
    'lat',
    'lon',
    'pop',
    'pop_dens',
    'frg_pct',
    'emp',
    'mean_taxable_income',
    'dist_supermarket',
]

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Fit and transform the features
df[features_to_scale] = scaler.fit_transform(df[features_to_scale])


# %%NBQA-CELL-SEP42d9ce
# Create train and test samples
X_train, X_test, y_train, y_test = train_test_split(
    df[features_to_scale], df['price'], test_size=0.20, random_state=42
)

# Show X_train
print('X_train:')
print(X_train.head(), '\n')

# Show y_train
print('y_train:')
print(y_train.head())


# %%NBQA-CELL-SEP42d9ce
# Define the number of features
num_features = X_train.shape[1]

# Define the model
model = keras.Sequential(
    [
        keras.Input(shape=(num_features,)),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1),
    ]
)

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mape'])

# Train the model
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    validation_split=0.20,
    batch_size=32,
    verbose=0,
)

# Predict the response for test dataset
y_pred = model.predict(X_test)

# Evaluate the model on the test set using the mean absolute error (MAPE)
test_loss, test_mape = model.evaluate(X_test, y_test)
print(f"\nMAPE: {test_mape:.2f}")

# Calculate R2 score
r2 = r2_score(y_test, y_pred)
print(f"R2 score: {r2:.4f}")


# %%NBQA-CELL-SEP42d9ce
# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

# Plot training & validation MAE values
plt.plot(history.history['mape'])
plt.plot(history.history['val_mape'])
plt.title('Model MAPE')
plt.ylabel('MAPE')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()


# %%NBQA-CELL-SEP42d9ce
# Show operating system, date/time and Python version
print('-----------------------------------')
print(os.name.upper())
print(platform.system(), '|', platform.release())
print('Datetime:', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print('Python Version:', python_version())
print('-----------------------------------')
