# 💹 Financial Analysis Tool

An advanced, modular Python-based tool for comprehensive financial data analysis, prediction, and optimization. This project integrates financial modeling, natural language processing, portfolio optimization, and news-based sentiment analysis — all in one package.

---

## 🚀 Features

- 📈 **Stock Data Management**
  - Real-time and historical data retrieval
  - Price visualization and technical indicators

- 📰 **News Sentiment Analysis**
  - Fetches financial news
  - Applies NLP techniques to extract and evaluate sentiment impact on assets

- 🤖 **Machine Learning Modeling**
  - Predictive models for stock prices and trends
  - Model selection and training pipeline

- 📊 **Portfolio Optimization**
  - Efficient frontier plotting
  - Maximizes Sharpe ratio and minimizes risk

- ⚙️ **Modular Structure**
  - Clean, decoupled architecture for maintainability and scalability

---

## 🧠 Technologies Used

- Python 3.x
- `pandas`, `numpy`, `yfinance` – data wrangling
- `scikit-learn`, `xgboost` – machine learning
- `matplotlib`, `seaborn` – visualization
- `nltk`, `spacy`, `transformers` – NLP
- `cvxpy`, `PyPortfolioOpt` – portfolio optimization

---

## 📁 Project Structure

Financial_Analysis_Tool/

│  
├── app.py # Main app entry point  
├── config.py # Configuration settings  
├── requirements.txt # Python dependencies  
│  
├── managers/  
│ ├── stock_manager.py # Handles stock data retrieval  
│ ├── news_manager.py # Fetches & processes financial news  
│ ├── nlp_manager.py # Performs sentiment analysis  
│ ├── modeling_manager.py # Builds & evaluates ML models  
│ └── portfolio_optimizer.py # Runs portfolio optimization  
