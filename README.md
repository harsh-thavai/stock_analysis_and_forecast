# Stock Analysis and Forecast

## Project Overview
This project is an interactive web application for stock analysis and forecasting. It allows users to input a stock ticker and date range, then provides detailed analysis including price movements, key metrics, future price predictions, and relevant news.

demo link https://youtu.be/kM8hfslRvrA?si=3rQYhb26EgRt-EYQ
![b3f6ffc214886e86d998b5e6b1305f5f33cd34ffd22f520cec471fcd](https://github.com/harsh-thavai/stock_analysis_and_forecast/assets/114677475/cd44aa6c-b51e-4db6-9b8f-4342027260da)
![newplot (1)](https://github.com/harsh-thavai/stock_analysis_and_forecast/assets/114677475/d9285254-3435-4fe5-9c7d-d9d485d36ac5)

## Features
- Stock price visualization with candlestick charts
- Calculation and display of key financial metrics
- Price prediction using Facebook's Prophet model
- Display of top news related to the selected stock
- Interactive UI built with Streamlit

## Technologies Used
- Python 3.8+
- Streamlit for the web interface
- yfinance for fetching stock data
- Pandas and NumPy for data manipulation
- Plotly for interactive charts
- Facebook Prophet for time series forecasting
- StockNews for fetching stock-related news

## Installation & Usage

1. Clone the repository:
```git clone https://github.com/harsh-thavai/stock_analysis_and_forecast.git```
2. Install the required packages:
```pip install -r requirements.txt```
3. Run the Streamlit app:
```streamlit run app.py```
4. Open a web browser and go to
```http://localhost:8501```
5. Enter a stock ticker (e.g., AAPL for Apple Inc.) and select a date range
6. Explore the various tabs for different analyses and visualizations

## Detailed Component Descriptions

### Data Fetching and Processing
- Stock data is fetched using the `yfinance` library
- Additional metrics like daily returns and cumulative returns are calculated

### Visualization
- Interactive candlestick charts for stock prices
- Bar charts for daily returns
- Line charts for cumulative returns

### Metrics Calculation
- Total Return
- Annual Return
- Annual Standard Deviation
- Sharpe Ratio

### Price Prediction
- Utilizes Facebook's Prophet model for time series forecasting
- Allows user to select prediction timeframe (1-4 years)
- Displays forecast plot and forecast components

### News Integration
- Fetches top news related to the selected stock using the `stocknews` library

## Future Improvements

- [ ] Integrate with a database for data storage
- [ ] Implement cloud deployment (e.g., AWS, Azure, GCP)
- [ ] Add comprehensive logging using Python's logging library
- [ ] Integrate with MLflow for experiment tracking and model versioning
- [ ] Implement additional data sources and web scraping for more comprehensive analysis
- [ ] Add unit and integration tests
- [ ] Optimize model performance and code efficiency

## Contributing
Contributions to this project are welcome! Please fork the repository and submit a pull request with your changes.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
For any queries or suggestions, please open an issue on this GitHub repository.
   
