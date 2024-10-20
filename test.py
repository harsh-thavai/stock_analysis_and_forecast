import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import date, timedelta
from prophet import Prophet
from prophet.plot import plot_plotly
from textblob import TextBlob
import ta
import warnings

warnings.filterwarnings('ignore')

# Function to calculate Garman-Klass Volatility
def garman_klass_volatility(data, window=30):
    log_hl = (data['High'] / data['Low']).apply(np.log)
    log_co = (data['Close'] / data['Open']).apply(np.log)
    
    rs = 0.5 * log_hl**2 - (2*np.log(2)-1) * log_co**2
    
    return np.sqrt(rs.rolling(window=window).mean() * 252)

# Function to add metrics and technical indicators to dataframe
def add_metrics_and_indicators(df):
    df['daily_return'] = df['Adj Close'].pct_change()
    df['cumulative_return'] = (1 + df['daily_return']).cumprod() - 1
    
    # Add technical indicators
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df['MACD'] = ta.trend.macd_diff(df['Close'])
    df['BB_high'] = ta.volatility.bollinger_hband(df['Close'])
    df['BB_low'] = ta.volatility.bollinger_lband(df['Close'])
    
    # Add Garman-Klass Volatility
    df['GK_Volatility'] = garman_klass_volatility(df)
    
    trading_days = 252
    total_return = df['cumulative_return'].iloc[-1]
    annual_return = (1 + total_return) ** (trading_days / len(df)) - 1
    daily_std = df['daily_return'].std()
    annual_std = daily_std * np.sqrt(trading_days)
    sharpe_ratio = (annual_return - 0.02) / annual_std

    return df, {
        'Total Return': total_return,
        'Annual Return': annual_return,
        'Annual Std Dev': annual_std,
        'Sharpe Ratio': sharpe_ratio
    }

# Function for forecasting using Prophet
@st.cache_resource(ttl=3600)
def prophet_forecast(data, periods):
    df = data.reset_index()[['Date', 'Close']]
    df.columns = ['ds', 'y']
    model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return model, forecast

# Function for sentiment analysis
def analyze_sentiment(text):
    return TextBlob(text).sentiment.polarity

# Streamlit app
def main():
    st.set_page_config(layout="wide")
    st.title('📈 Advanced Stock Analysis and Forecast Dashboard')

    # Add GitHub and LinkedIn links
    st.markdown("""
    Created by Harsh Thavai | 
    [LinkedIn](https://www.linkedin.com/in/harsh-thavai/)
    """)

    # Sidebar for user input
    st.sidebar.header('📊 User Input')
    ticker = st.sidebar.text_input('Ticker Symbol', 'AAPL')
    start_date = st.sidebar.date_input('Start Date', date.today() - timedelta(days=365*5))  # 5 years of data
    end_date = st.sidebar.date_input('End Date', date.today())

    if ticker:
        try:
            @st.cache_data(ttl=3600)
            def load_data(ticker, start, end):
                return yf.download(ticker, start=start, end=end)

            data = load_data(ticker, start_date, end_date)

            if not data.empty:
                data, metrics = add_metrics_and_indicators(data)

                # Main content
                st.subheader(f"{ticker} Stock Overview")
                col1, col2, col3, col4 = st.columns(4)
                cols = [col1, col2, col3, col4]
                for i, (metric, value) in enumerate(metrics.items()):
                    cols[i].metric(metric, f"{value:.2%}", delta_color="normal")
                    
                # Tabs for different analyses
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Price & Volume", "📈 Technical Indicators", "📉 Returns Analysis", "🔮 Forecast", "📰 Market Sentiment"])

                with tab1:
                    st.subheader(f"{ticker} Stock Price and Volume")
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])

                    fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name="OHLC"), row=1, col=1)

                    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name="Volume", marker_color='rgba(0, 0, 255, 0.5)'), row=2, col=1)

                    fig.update_layout(height=600, title_text="Candlestick Chart with Volume", showlegend=False)
                    fig.update_xaxes(title_text="Date", row=2, col=1)
                    fig.update_yaxes(title_text="Price", row=1, col=1)
                    fig.update_yaxes(title_text="Volume", row=2, col=1)
                    st.plotly_chart(fig, use_container_width=True)

                    # Add data table
                    st.subheader("Recent Data")
                    st.dataframe(data.tail(10).style.highlight_max(axis=0))

                with tab2:
                    st.subheader('Technical Indicators')
                    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, 
                                        vertical_spacing=0.05, row_heights=[0.4, 0.2, 0.2, 0.2])

                    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close Price', line=dict(color='blue')), row=1, col=1)
                    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], name='SMA 20', line=dict(color='orange')), row=1, col=1)
                    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], name='SMA 50', line=dict(color='red')), row=1, col=1)
                    fig.add_trace(go.Scatter(x=data.index, y=data['BB_high'], name='BB High', line=dict(color='green', dash='dash')), row=1, col=1)
                    fig.add_trace(go.Scatter(x=data.index, y=data['BB_low'], name='BB Low', line=dict(color='green', dash='dash')), row=1, col=1)

                    fig.add_trace(go.Bar(x=data.index, y=data['MACD'], name='MACD', marker_color='purple'), row=2, col=1)

                    fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='teal')), row=3, col=1)
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

                    fig.add_trace(go.Scatter(x=data.index, y=data['GK_Volatility'], name='GK Volatility', line=dict(color='orange')), row=4, col=1)

                    fig.update_layout(height=1000, title_text="Technical Indicators", showlegend=True)
                    fig.update_xaxes(title_text="Date", row=4, col=1)
                    fig.update_yaxes(title_text="Price", row=1, col=1)
                    fig.update_yaxes(title_text="MACD", row=2, col=1)
                    fig.update_yaxes(title_text="RSI", row=3, col=1)
                    fig.update_yaxes(title_text="GK Volatility", row=4, col=1)
                    st.plotly_chart(fig, use_container_width=True)

                with tab3:
                    st.subheader('Returns Analysis')
                    fig_returns = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                                vertical_spacing=0.03, subplot_titles=('Daily Returns', 'Cumulative Returns'))

                    fig_returns.add_trace(go.Bar(x=data.index, y=data['daily_return'], 
                                                 marker_color=np.where(data['daily_return'] < 0, 'red', 'green'),
                                                 name='Daily Returns'), row=1, col=1)

                    fig_returns.add_trace(go.Scatter(x=data.index, y=data['cumulative_return'], 
                                                     line=dict(color='blue', width=2),
                                                     name='Cumulative Returns'), row=2, col=1)

                    fig_returns.update_layout(height=500, title_text="Returns Analysis", showlegend=True)
                    fig_returns.update_xaxes(title_text="Date", row=2, col=1)
                    fig_returns.update_yaxes(title_text="Daily Return", row=1, col=1)
                    fig_returns.update_yaxes(title_text="Cumulative Return", row=2, col=1)
                    st.plotly_chart(fig_returns, use_container_width=True)

                    # Correlation heatmap
                    st.subheader('Correlation Heatmap')
                    corr_matrix = data[['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'GK_Volatility']].corr()
                    fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
                    fig_corr.update_layout(title_text="Correlation Heatmap")
                    st.plotly_chart(fig_corr, use_container_width=True)

                with tab4:
                    st.subheader('Prophet Forecast')
                    forecast_days = st.slider('Forecast Days', 30, 1825, 365)  # Up to 5 years forecast
                    
                    with st.spinner('Generating forecast...'):
                        model, forecast = prophet_forecast(data, forecast_days)
                    
                    fig_forecast = plot_plotly(model, forecast)
                    fig_forecast.update_layout(title_text="Prophet Price Forecast", xaxis_title="Date", yaxis_title="Price")
                    st.plotly_chart(fig_forecast, use_container_width=True)

                    # Add forecast evaluation metrics
                    last_price = data['Close'].iloc[-1]
                    forecast_price = forecast.iloc[-1]['yhat']
                    forecast_change = (forecast_price - last_price) / last_price
                    st.metric("Forecasted Price Change", f"{forecast_change:.2%}", 
                              delta=f"{forecast_price:.2f}", delta_color="normal")

                    # Show forecast components
                    st.subheader("Forecast Components")
                    fig_components = model.plot_components(forecast)
                    st.pyplot(fig_components)

                with tab5:
                    st.subheader('Market Sentiment Analysis')
                    
                    @st.cache_data(ttl=3600) 
                    def get_news(ticker):
                        try:
                            stock = yf.Ticker(ticker)
                            return stock.news
                        except Exception as e:
                            st.error(f"Error fetching news: {str(e)}")
                            return []

                    news = get_news(ticker)
                    if news:
                        sentiments = []
                        for article in news[:5]:  # Display top 5 news articles
                            try:
                                title = article.get('title', 'No title available')
                                sentiment = analyze_sentiment(title)
                                sentiments.append(sentiment)
                                sentiment_color = 'green' if sentiment > 0 else 'red' if sentiment < 0 else 'gray'
                                st.markdown(f"**{title}**")
                                st.markdown(f"Sentiment: <span style='color:{sentiment_color}'>{sentiment:.2f}</span>", unsafe_allow_html=True)
                                st.write(f"Published: {pd.to_datetime(article.get('providerPublishTime', 0), unit='s')}")
                                st.write(article.get('link', 'No link available'))
                                st.markdown("---")
                            except Exception as e:
                                st.error(f"Error processing news article: {str(e)}")

                        if sentiments:
                            # Overall sentiment analysis
                            overall_sentiment = np.mean(sentiments)
                            st.subheader('Overall Market Sentiment')
                            fig_sentiment = go.Figure(go.Indicator(
                                mode = "gauge+number",
                                value = overall_sentiment,
                                domain = {'x': [0, 1], 'y': [0, 1]},
                                title = {'text': "Sentiment", 'font': {'size': 24}},
                                gauge = {
                                    'axis': {'range': [-1, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
                                    'bar': {'color': "darkblue"},
                                    'bgcolor': "white",
                                    'borderwidth': 2,
                                    'bordercolor': "gray",
                                    'steps': [
                                        {'range': [-1, -0.5], 'color': 'red'},
                                        {'range': [-0.5, 0.5], 'color': 'gray'},
                                        {'range': [0.5, 1], 'color': 'green'}],
                                    'threshold': {
                                        'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75,
                                        'value': overall_sentiment}}))
                            fig_sentiment.update_layout(height=300)
                            st.plotly_chart(fig_sentiment, use_container_width=True)
                        else:
                            st.warning("No sentiment data available.")
                    else:
                        st.warning("No news articles found for this stock.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("If the error persists, please check your input and try again.")
    else:
        st.info("Please enter a ticker symbol to start.")

if __name__ == "__main__":
    main()
