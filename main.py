# Import necessary libraries
import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from stocknews import StockNews
from datetime import date
from prophet import Prophet
from prophet.plot import plot_plotly
import numpy as np
import warnings




def add_metrics_to_df(df):
    """
    Calculate and add financial metrics to the dataframe.
    
    Args:
    df (pd.DataFrame): Input dataframe with stock price data.
    
    Returns:
    tuple: Updated dataframe and a dictionary of calculated metrics.
    """
    # Calculate daily return
    df['daily_return'] = (df['Adj Close'] / df['Adj Close'].shift(1)) - 1
    
    # Calculate cumulative return
    df['cumulative_return'] = (1 + df['daily_return']).cumprod() - 1
    
    # Calculate financial metrics
    trading_days = 252  # Approximate number of trading days in a year
    
    total_return = df['cumulative_return'].iloc[-1]
    annual_return = (1 + total_return) ** (trading_days / len(df)) - 1
    
    daily_std = df['daily_return'].std()
    annual_std = daily_std * np.sqrt(trading_days)
    
    sharpe_ratio = (annual_return - 0.02) / annual_std  # Assuming 2% risk-free rate
    
    metrics = {
        'Total Return': total_return,
        'Annual Return': annual_return,
        'Annual Std Dev': annual_std,
        'Sharpe Ratio': sharpe_ratio
    }
    
    return df, metrics

# Set up Streamlit dashboard
st.title('Stock ANALYSIS AND FORECAST')

# Sidebar for user input
st.sidebar.header('User Input')
ticker = st.sidebar.text_input('Ticker')
start_date = st.sidebar.date_input('Start Date')
end_date = st.sidebar.date_input('End Date')

if ticker:
    try:
        # Fetch stock data
        data = yf.download(ticker, start=start_date, end=end_date)
                
        if not data.empty:
            # Add metrics to the dataframe
            data, metrics = add_metrics_to_df(data)
            
            # Create candlestick chart
            fig = go.Figure(data=[go.Candlestick(x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'])])
            fig.update_layout(title=f"{ticker} Stock Price", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig)

            # Create tabs for different analyses
            pricing_data, metrics_tab, prediction, news = st.tabs(["Pricing Data", "Metrics", "Prediction", "TOP NEWS"])

            with pricing_data:
                st.header('Price Movements')
                
                def color_negative_red(val):
                    """Color negative values red and positive values green."""
                    color = 'red' if val < 0 else 'green'
                    return f'color: {color}'
                
                # Display styled dataframe
                styled_data = data.style.applymap(color_negative_red, subset=['daily_return', 'cumulative_return'])
                st.dataframe(styled_data)
                
                # Plot daily returns
                fig_returns = go.Figure()
                fig_returns.add_trace(go.Bar(
                    x=data.index, 
                    y=data['daily_return'],
                    name='Daily Returns',
                    marker_color=['red' if ret < 0 else 'green' for ret in data['daily_return']]
                ))
                fig_returns.update_layout(title=f"{ticker} Daily Returns", xaxis_title='Date', yaxis_title='Return')
                st.plotly_chart(fig_returns)
                
                # Plot cumulative returns
                fig_cum_returns = go.Figure()
                fig_cum_returns.add_trace(go.Scatter(x=data.index, y=data['cumulative_return'], mode='lines', name='Cumulative Returns'))
                fig_cum_returns.update_layout(title=f"{ticker} Cumulative Returns", xaxis_title='Date', yaxis_title='Cumulative Return')
                st.plotly_chart(fig_cum_returns)

            with metrics_tab:
                st.header('Key Metrics')
                # Display calculated metrics
                for metric, value in metrics.items():
                    st.metric(metric, f"{value:.2%}")
            
            with prediction:
                st.header('Prediction')
                
                # Set default dates for prediction
                START = "2015-01-01"
                TODAY = date.today().strftime("%Y-%m-%d")
                
                # Allow user to select prediction timeframe
                n_years = st.slider('Years of prediction:', 1, 4)
                period = n_years * 365

                @st.cache_data
                def load_data(ticker):
                    """Load and cache stock data."""
                    data = yf.download(ticker, START, TODAY)
                    data.reset_index(inplace=True)
                    return data

                # Load data for prediction
                data_load_state = st.text('Loading data...')
                data = load_data(ticker)
                data_load_state.text('Loading data... done!')

                # Display raw data
                st.subheader('Raw data')
                st.write(data.tail())

                def plot_raw_data():
                    """Plot raw stock price data."""
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
                    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
                    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
                    st.plotly_chart(fig)

                plot_raw_data()

                # Predict forecast with Prophet
                df_train = data[['Date','Close']]
                df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
                m = Prophet()
                m.fit(df_train)
                future = m.make_future_dataframe(periods=period)
                forecast = m.predict(future)

                # Show and plot forecast
                st.subheader('Forecast data')
                st.write(forecast.tail())
                
                st.write(f'Forecast plot for {n_years} years')
                fig1 = plot_plotly(m, forecast)
                st.plotly_chart(fig1)

                st.write("Forecast components")
                fig2 = m.plot_components(forecast)
                st.pyplot(fig2)

            with news:
                st.header('TOP NEWS')
                
                # Fetch news using stocknews
                sn = StockNews(ticker, save_news=False)
                df_news = sn.read_rss()
                
                # Display top 10 news items
                for i in range(min(10, len(df_news))):
                    st.subheader(f"News {i+1}")
                    st.write(f"Published Date: {df_news['published'][i]}")
                    st.write(f"Title: {df_news['title'][i]}")
                    
                    if 'summary' in df_news.columns:
                        st.write(f"Summary: {df_news['summary'][i]}")
                    elif 'text' in df_news.columns:
                        st.write(f"Text: {df_news['text'][i][:500]}...")  # Display first 500 characters
                    
                    st.write("---")

        else:
            st.error("No data available for the selected ticker and date range.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.error(f"Error details: {str(e)}")
else:
    st.info("Please enter a ticker symbol to start.")
