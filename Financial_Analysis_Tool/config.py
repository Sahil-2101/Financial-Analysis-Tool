"""
Configuration file for the Finance Analytical Tool.
Contains API keys and other configuration variables.
"""

# OpenAI API Configuration
OPENAI_API_KEY = "sk-efghijklmnop5678efghijklmnop5678efghijkl"

# Model Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Stock Data Configuration
DEFAULT_START_DATE = "2020-01-01"
DEFAULT_END_DATE = None  # Will be set to current date in the application

# News Configuration
DEFAULT_NUM_HEADLINES = 5

# Model Parameters
RANDOM_FOREST_PARAMS = {
    "n_estimators": 100,
    "random_state": 42
}

SVR_PARAMS = {
    "kernel": 'rbf',
    "C": 100,
    "gamma": 0.1,
    "epsilon": 0.1
}

NEURAL_NETWORK_PARAMS = {
    "epochs": 50,
    "batch_size": 32
} 