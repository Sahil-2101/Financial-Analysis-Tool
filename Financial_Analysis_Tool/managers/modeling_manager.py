"""
Modeling management module for stock price prediction using various machine learning models.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from config import (
    RANDOM_STATE,
    TEST_SIZE,
    RANDOM_FOREST_PARAMS,
    SVR_PARAMS,
    NEURAL_NETWORK_PARAMS
)

class ModelingManager:
    @staticmethod
    def prepare_stock_data_for_modeling(stock_data, target_column="Close", test_size=TEST_SIZE, random_state=RANDOM_STATE):
        """
        Prepares stock data for modeling by using the date index as a feature and a specified column as the target.

        Args:
            stock_data (pandas.DataFrame): Stock data with a datetime index and target column.
            target_column (str): The column to use as the target variable.
            test_size (float): Proportion of data to use for testing.
            random_state (int): Random seed for reproducibility.

        Returns:
            tuple: X_train, X_test, y_train, y_test - Training and testing datasets.
        """
        # Ensure the index is datetime
        if not np.issubdtype(stock_data.index.dtype, np.datetime64):
            raise ValueError("The stock data index must be a datetime type.")

        # Use date index as a numerical feature
        X = np.arange(len(stock_data)).reshape(-1, 1)  # Numerical representation of time
        y = stock_data[target_column].values  # Target values (e.g., closing prices)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        return X_train, X_test, y_train, y_test

    @staticmethod
    def random_forest_model(X_train, y_train, X_test):
        model = RandomForestRegressor(**RANDOM_FOREST_PARAMS)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return model, y_pred

    @staticmethod
    def decision_tree_model(X_train, y_train, X_test):
        model = DecisionTreeRegressor(random_state=RANDOM_STATE)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return model, y_pred

    @staticmethod
    def svr_model(X_train, y_train, X_test):
        model = SVR(**SVR_PARAMS)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return model, y_pred

    @staticmethod
    def neural_network_model(X_train, y_train, X_test):
        model = Sequential([
            Dense(64, activation='relu', input_dim=1),
            Dense(32, activation='relu'),
            Dense(1)  # Output layer
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=NEURAL_NETWORK_PARAMS['epochs'], 
                 batch_size=NEURAL_NETWORK_PARAMS['batch_size'], verbose=0)
        y_pred = model.predict(X_test).flatten()
        return model, y_pred 