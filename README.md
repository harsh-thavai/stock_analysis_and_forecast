# Advanced Stock Analysis and Forecast Dashboard

## 📊 Overview

This project is an advanced stock analysis and forecast dashboard built with Streamlit, offering a comprehensive suite of tools for in-depth stock market analysis. It provides real-time data visualization, technical indicators, returns analysis, price forecasting, and market sentiment analysis, all in one interactive web application.

![newplot(1)](https://github.com/user-attachments/assets/5d2089f8-4ff1-4fb2-a2e4-61b22a7dba50)


## 🌟 Features

- **Real-time Stock Data**: Fetch and display up-to-date stock information using yfinance.
- **Interactive Visualizations**: Utilize Plotly for creating dynamic and interactive charts.
- **Technical Analysis**: Implement various technical indicators including SMA, RSI, MACD, and Bollinger Bands.
- **Advanced Metrics**: Calculate and display key financial metrics such as Sharpe Ratio and Garman-Klass Volatility.
- **Returns Analysis**: Visualize daily and cumulative returns with interactive charts.
- **Price Forecasting**: Implement Facebook Prophet for time series forecasting of stock prices.
- **Sentiment Analysis**: Analyze market sentiment using latest news articles related to the stock.
- **User-friendly Interface**: Easy-to-use Streamlit interface with multiple tabs for different analyses.



## 🛠️ Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- yfinance
- Plotly
- Facebook Prophet
- TextBlob
- TA-Lib

## 📋 Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.7+
- pip package manager

## 🚀 Installation and Setup

1. Clone the repository:
   ```
   git clone [https://github.com/harsh-thavai/stock-analysis-dashboard.git](https://github.com/harsh-thavai/stock_analysis_and_forecast.git)
   cd stock-analysis-dashboard
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

5. Open your web browser and navigate to `http://localhost:8501` to view the dashboard.

## 🖥️ Usage

1. Enter a stock ticker symbol in the sidebar (e.g., AAPL for Apple Inc.).
2. Select the date range for analysis.
3. Explore different tabs for various analyses:
   - Price & Volume: View candlestick chart and volume data.
   - Technical Indicators: Analyze SMA, RSI, MACD, and Bollinger Bands.
   - Returns Analysis: Examine daily and cumulative returns.
   - Forecast: View price predictions using Prophet.
   - Market Sentiment: Analyze current market sentiment based on news.

## 📊 Dashboard Sections

### 1. Stock Overview
Displays key metrics including Total Return, Annual Return, Annual Standard Deviation, and Sharpe Ratio.

### 2. Price & Volume
- Interactive candlestick chart
- Volume bar chart
- Recent data table

### 3. Technical Indicators
- Simple Moving Averages (20 and 50 days)
- Relative Strength Index (RSI)
- Moving Average Convergence Divergence (MACD)
- Bollinger Bands
- Garman-Klass Volatility

### 4. Returns Analysis
- Daily returns bar chart
- Cumulative returns line chart
- Correlation heatmap of various metrics

### 5. Forecast
- Prophet-based price forecast
- Forecast components breakdown
- Customizable forecast period

### 6. Market Sentiment
- Latest news articles related to the stock
- Sentiment analysis of news headlines
- Overall market sentiment gauge

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check [issues page]([https://github.com/yourusername/stock-analysis-dashboard/issues](https://github.com/harsh-thavai/stock_analysis_and_forecast/issues)) if you want to contribute.

## 📝 License

This project is [MIT](https://choosealicense.com/licenses/mit/) licensed.

## 👏 Acknowledgements

- [Streamlit](https://streamlit.io/) for the amazing web app framework
- [yfinance](https://github.com/ranaroussi/yfinance) for providing easy access to Yahoo Finance data
- [Plotly](https://plotly.com/) for interactive plotting capabilities
- [Facebook Prophet](https://facebook.github.io/prophet/) for time series forecasting
