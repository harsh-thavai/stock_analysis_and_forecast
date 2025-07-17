import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import date, timedelta
import warnings

warnings.filterwarnings('ignore')

# Check if optional libraries are available
try:
    from prophet import Prophet
    from prophet.plot import plot_plotly
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    st.warning("Prophet library not installed. Forecasting functionality will be disabled.")

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    st.warning("TextBlob library not installed. Sentiment analysis will be disabled.")

try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    st.warning("TA library not installed. Technical indicators will be limited.")

# Function to calculate Garman-Klass Volatility
def garman_klass_volatility(data, window=30):
    try:
        log_hl = (data['High'] / data['Low']).apply(np.log)
        log_co = (data['Close'] / data['Open']).apply(np.log)
        
        rs = 0.5 * log_hl**2 - (2*np.log(2)-1) * log_co**2
        
        return np.sqrt(rs.rolling(window=window).mean() * 252)
    except Exception as e:
        st.error(f"Error calculating Garman-Klass volatility: {str(e)}")
        return pd.Series(index=data.index, dtype=float)

# Function to add metrics and technical indicators to dataframe
def add_metrics_and_indicators(df):
    try:
        df['daily_return'] = df['Adj Close'].pct_change()
        df['cumulative_return'] = (1 + df['daily_return']).cumprod() - 1
        
        # Add basic technical indicators (manual calculation if TA not available)
        if TA_AVAILABLE:
            df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
            df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
            df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
            df['MACD'] = ta.trend.macd_diff(df['Close'])
            df['BB_high'] = ta.volatility.bollinger_hband(df['Close'])
            df['BB_low'] = ta.volatility.bollinger_lband(df['Close'])
        else:
            # Manual calculation of basic indicators
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            
            # Simple RSI calculation
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Simple MACD
            ema_12 = df['Close'].ewm(span=12).mean()
            ema_26 = df['Close'].ewm(span=26).mean()
            df['MACD'] = ema_12 - ema_26
            
            # Simple Bollinger Bands
            sma_20 = df['Close'].rolling(window=20).mean()
            std_20 = df['Close'].rolling(window=20).std()
            df['BB_high'] = sma_20 + (std_20 * 2)
            df['BB_low'] = sma_20 - (std_20 * 2)
        
        # Add Garman-Klass Volatility
        df['GK_Volatility'] = garman_klass_volatility(df)
        
        # Calculate performance metrics
        trading_days = 252
        total_return = df['cumulative_return'].iloc[-1]
        annual_return = (1 + total_return) ** (trading_days / len(df)) - 1
        daily_std = df['daily_return'].std()
        annual_std = daily_std * np.sqrt(trading_days)
        sharpe_ratio = (annual_return - 0.02) / annual_std if annual_std != 0 else 0

        return df, {
            'Total Return': total_return,
            'Annual Return': annual_return,
            'Annual Std Dev': annual_std,
            'Sharpe Ratio': sharpe_ratio
        }
    except Exception as e:
        st.error(f"Error calculating metrics: {str(e)}")
        return df, {}

# Function for forecasting using Prophet
def prophet_forecast(data, periods):
    if not PROPHET_AVAILABLE:
        st.error("Prophet library not available for forecasting.")
        return None, None
    
    try:
        df = data.reset_index()[['Date', 'Close']].copy()
        df.columns = ['ds', 'y']
        df = df.dropna()
        
        model = Prophet(
            daily_seasonality=True, 
            weekly_seasonality=True, 
            yearly_seasonality=True,
            changepoint_prior_scale=0.05
        )
        model.fit(df)
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        return model, forecast
    except Exception as e:
        st.error(f"Error in Prophet forecasting: {str(e)}")
        return None, None

# Function for sentiment analysis
def analyze_sentiment(text):
    if not TEXTBLOB_AVAILABLE:
        return 0
    try:
        return TextBlob(text).sentiment.polarity
    except Exception as e:
        st.error(f"Error in sentiment analysis: {str(e)}")
        return 0

def test_yfinance_connection():
    """Test if yfinance is working properly"""
    try:
        # Test with a simple ticker
        test_ticker = yf.Ticker('AAPL')
        test_data = test_ticker.history(period='5d')
        return not test_data.empty, test_data
    except Exception as e:
        return False, str(e)

# Streamlit app
def main():
    st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")
    st.title('📈 Advanced Stock Analysis and Forecast Dashboard')

    # Add GitHub and LinkedIn links
    st.markdown("""
    Created by Harsh Thavai | 
    [LinkedIn](https://www.linkedin.com/in/harsh-thavai/)
    """)
    
    # Test yfinance connection on startup
    with st.spinner('Testing connection to Yahoo Finance...'):
        connection_ok, test_result = test_yfinance_connection()
        if connection_ok:
            st.success("✅ Successfully connected to Yahoo Finance!")
        else:
            st.error(f"❌ Cannot connect to Yahoo Finance: {test_result}")
            st.error("Please check your internet connection and try again.")
            return

    # Sidebar for user input
    st.sidebar.header('📊 User Input')
    
    # Add some example tickers
    st.sidebar.write("**Popular tickers:** AAPL, GOOGL, MSFT, TSLA, AMZN, NVDA")
    
    ticker = st.sidebar.text_input('Ticker Symbol', 'AAPL').upper().strip()
    start_date = st.sidebar.date_input('Start Date', date.today() - timedelta(days=365))  # 1 year default
    end_date = st.sidebar.date_input('End Date', date.today())
    
    # Add a test connection button
    if st.sidebar.button('Test Connection'):
        try:
            test_ticker = yf.Ticker('AAPL')
            test_info = test_ticker.info
            st.sidebar.success("✅ Connection to Yahoo Finance is working!")
            st.sidebar.write(f"Test result: {test_info.get('longName', 'Apple Inc.')}")
        except Exception as e:
            st.sidebar.error(f"❌ Connection failed: {str(e)}")
            st.sidebar.error("Try refreshing the page or checking your internet connection")

    # Validate dates
    if start_date >= end_date:
        st.error("Start date must be before end date.")
        return

    if ticker:
        try:
            @st.cache_data(ttl=3600)
            def load_data(ticker, start, end):
                try:
                    # Try different approaches to download data
                    st.info(f"Attempting to download data for {ticker}...")
                    
                    # Method 1: Standard download
                    data = yf.download(ticker, start=start, end=end, progress=False, show_errors=False)
                    
                    if data.empty:
                        st.info("Trying alternative download method...")
                        # Method 2: Using Ticker object
                        ticker_obj = yf.Ticker(ticker)
                        data = ticker_obj.history(start=start, end=end)
                    
                    if data.empty:
                        st.info("Trying with different date range...")
                        # Method 3: Try with a shorter date range
                        recent_start = end - timedelta(days=365)
                        data = yf.download(ticker, start=recent_start, end=end, progress=False, show_errors=False)
                    
                    if not data.empty:
                        st.success(f"Successfully downloaded {len(data)} days of data for {ticker}")
                        # Print data info for debugging
                        st.write(f"Data range: {data.index.min()} to {data.index.max()}")
                        st.write(f"Columns: {list(data.columns)}")
                    
                    return data
                    
                except Exception as e:
                    st.error(f"Error downloading data for {ticker}: {str(e)}")
                    st.error("Possible solutions:")
                    st.error("1. Check if the ticker symbol is correct")
                    st.error("2. Try a different date range")
                    st.error("3. Check your internet connection")
                    st.error("4. Yahoo Finance might be temporarily unavailable")
                    return pd.DataFrame()

            with st.spinner(f'Loading data for {ticker}...'):
                data = load_data(ticker, start_date, end_date)

            if not data.empty and len(data) > 0:
                # Reset index to have Date as a column
                data = data.reset_index()
                data, metrics = add_metrics_and_indicators(data)

                # Main content
                st.subheader(f"{ticker} Stock Overview")
                if metrics:
                    col1, col2, col3, col4 = st.columns(4)
                    cols = [col1, col2, col3, col4]
                    for i, (metric, value) in enumerate(metrics.items()):
                        if not pd.isna(value):
                            cols[i].metric(metric, f"{value:.2%}")
                        else:
                            cols[i].metric(metric, "N/A")
                    
                # Tabs for different analyses
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Price & Volume", "📈 Technical Indicators", "📉 Returns Analysis", "🔮 Forecast", "📰 Market Sentiment"])

                with tab1:
                    st.subheader(f"{ticker} Stock Price and Volume")
                    
                    fig = make_subplots(
                        rows=2, cols=1, 
                        shared_xaxes=True, 
                        vertical_spacing=0.03, 
                        row_heights=[0.7, 0.3]
                    )

                    # Add candlestick chart
                    fig.add_trace(go.Candlestick(
                        x=data['Date'], 
                        open=data['Open'], 
                        high=data['High'], 
                        low=data['Low'], 
                        close=data['Close'], 
                        name="OHLC"
                    ), row=1, col=1)

                    # Add volume chart
                    fig.add_trace(go.Bar(
                        x=data['Date'], 
                        y=data['Volume'], 
                        name="Volume", 
                        marker_color='rgba(0, 0, 255, 0.5)'
                    ), row=2, col=1)

                    fig.update_layout(
                        height=600, 
                        title_text="Candlestick Chart with Volume", 
                        showlegend=False
                    )
                    fig.update_xaxes(title_text="Date", row=2, col=1)
                    fig.update_yaxes(title_text="Price", row=1, col=1)
                    fig.update_yaxes(title_text="Volume", row=2, col=1)
                    st.plotly_chart(fig, use_container_width=True)

                    # Add data table
                    st.subheader("Recent Data")
                    st.dataframe(data.tail(10))

                with tab2:
                    st.subheader('Technical Indicators')
                    
                    fig = make_subplots(
                        rows=4, cols=1, 
                        shared_xaxes=True, 
                        vertical_spacing=0.05, 
                        row_heights=[0.4, 0.2, 0.2, 0.2]
                    )

                    # Price and moving averages
                    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Close Price', line=dict(color='blue')), row=1, col=1)
                    if 'SMA_20' in data.columns:
                        fig.add_trace(go.Scatter(x=data['Date'], y=data['SMA_20'], name='SMA 20', line=dict(color='orange')), row=1, col=1)
                    if 'SMA_50' in data.columns:
                        fig.add_trace(go.Scatter(x=data['Date'], y=data['SMA_50'], name='SMA 50', line=dict(color='red')), row=1, col=1)
                    if 'BB_high' in data.columns and 'BB_low' in data.columns:
                        fig.add_trace(go.Scatter(x=data['Date'], y=data['BB_high'], name='BB High', line=dict(color='green', dash='dash')), row=1, col=1)
                        fig.add_trace(go.Scatter(x=data['Date'], y=data['BB_low'], name='BB Low', line=dict(color='green', dash='dash')), row=1, col=1)

                    # MACD
                    if 'MACD' in data.columns:
                        fig.add_trace(go.Bar(x=data['Date'], y=data['MACD'], name='MACD', marker_color='purple'), row=2, col=1)

                    # RSI
                    if 'RSI' in data.columns:
                        fig.add_trace(go.Scatter(x=data['Date'], y=data['RSI'], name='RSI', line=dict(color='teal')), row=3, col=1)
                        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
                        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

                    # Volatility
                    if 'GK_Volatility' in data.columns:
                        fig.add_trace(go.Scatter(x=data['Date'], y=data['GK_Volatility'], name='GK Volatility', line=dict(color='orange')), row=4, col=1)

                    fig.update_layout(height=1000, title_text="Technical Indicators", showlegend=True)
                    fig.update_xaxes(title_text="Date", row=4, col=1)
                    fig.update_yaxes(title_text="Price", row=1, col=1)
                    fig.update_yaxes(title_text="MACD", row=2, col=1)
                    fig.update_yaxes(title_text="RSI", row=3, col=1)
                    fig.update_yaxes(title_text="GK Volatility", row=4, col=1)
                    st.plotly_chart(fig, use_container_width=True)

                with tab3:
                    st.subheader('Returns Analysis')
                    
                    if 'daily_return' in data.columns and 'cumulative_return' in data.columns:
                        fig_returns = make_subplots(
                            rows=2, cols=1, 
                            shared_xaxes=True, 
                            vertical_spacing=0.03, 
                            subplot_titles=('Daily Returns', 'Cumulative Returns')
                        )

                        # Daily returns
                        colors = ['red' if x < 0 else 'green' for x in data['daily_return']]
                        fig_returns.add_trace(go.Bar(
                            x=data['Date'], 
                            y=data['daily_return'], 
                            marker_color=colors,
                            name='Daily Returns'
                        ), row=1, col=1)

                        # Cumulative returns
                        fig_returns.add_trace(go.Scatter(
                            x=data['Date'], 
                            y=data['cumulative_return'], 
                            line=dict(color='blue', width=2),
                            name='Cumulative Returns'
                        ), row=2, col=1)

                        fig_returns.update_layout(height=500, title_text="Returns Analysis", showlegend=True)
                        fig_returns.update_xaxes(title_text="Date", row=2, col=1)
                        fig_returns.update_yaxes(title_text="Daily Return", row=1, col=1)
                        fig_returns.update_yaxes(title_text="Cumulative Return", row=2, col=1)
                        st.plotly_chart(fig_returns, use_container_width=True)

                        # Correlation heatmap
                        st.subheader('Correlation Heatmap')
                        corr_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                        available_columns = [col for col in corr_columns if col in data.columns]
                        
                        if len(available_columns) > 1:
                            corr_matrix = data[available_columns].corr()
                            fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
                            fig_corr.update_layout(title_text="Correlation Heatmap")
                            st.plotly_chart(fig_corr, use_container_width=True)

                with tab4:
                    st.subheader('Prophet Forecast')
                    
                    if PROPHET_AVAILABLE:
                        forecast_days = st.slider('Forecast Days', 30, 365, 90)
                        
                        if st.button('Generate Forecast'):
                            with st.spinner('Generating forecast...'):
                                model, forecast = prophet_forecast(data, forecast_days)
                            
                            if model is not None and forecast is not None:
                                fig_forecast = plot_plotly(model, forecast)
                                fig_forecast.update_layout(
                                    title_text="Prophet Price Forecast", 
                                    xaxis_title="Date", 
                                    yaxis_title="Price"
                                )
                                st.plotly_chart(fig_forecast, use_container_width=True)

                                # Forecast metrics
                                last_price = data['Close'].iloc[-1]
                                forecast_price = forecast.iloc[-1]['yhat']
                                forecast_change = (forecast_price - last_price) / last_price
                                
                                st.metric(
                                    "Forecasted Price Change", 
                                    f"{forecast_change:.2%}", 
                                    delta=f"${forecast_price:.2f}"
                                )
                    else:
                        st.error("Prophet library not available. Please install it with: pip install prophet")

                with tab5:
                    st.subheader('Market Sentiment Analysis')
                    
                    if TEXTBLOB_AVAILABLE:
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
                            for article in news[:5]:
                                try:
                                    title = article.get('title', 'No title available')
                                    sentiment = analyze_sentiment(title)
                                    sentiments.append(sentiment)
                                    sentiment_color = 'green' if sentiment > 0 else 'red' if sentiment < 0 else 'gray'
                                    
                                    st.markdown(f"**{title}**")
                                    st.markdown(f"Sentiment: <span style='color:{sentiment_color}'>{sentiment:.2f}</span>", unsafe_allow_html=True)
                                    
                                    publish_time = article.get('providerPublishTime', 0)
                                    if publish_time:
                                        st.write(f"Published: {pd.to_datetime(publish_time, unit='s')}")
                                    
                                    link = article.get('link', '')
                                    if link:
                                        st.write(f"[Read more]({link})")
                                    
                                    st.markdown("---")
                                except Exception as e:
                                    st.error(f"Error processing news article: {str(e)}")

                            if sentiments:
                                overall_sentiment = np.mean(sentiments)
                                st.subheader('Overall Market Sentiment')
                                
                                fig_sentiment = go.Figure(go.Indicator(
                                    mode="gauge+number",
                                    value=overall_sentiment,
                                    domain={'x': [0, 1], 'y': [0, 1]},
                                    title={'text': "Sentiment", 'font': {'size': 24}},
                                    gauge={
                                        'axis': {'range': [-1, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
                                        'bar': {'color': "darkblue"},
                                        'bgcolor': "white",
                                        'borderwidth': 2,
                                        'bordercolor': "gray",
                                        'steps': [
                                            {'range': [-1, -0.5], 'color': 'red'},
                                            {'range': [-0.5, 0.5], 'color': 'gray'},
                                            {'range': [0.5, 1], 'color': 'green'}
                                        ],
                                        'threshold': {
                                            'line': {'color': "red", 'width': 4},
                                            'thickness': 0.75,
                                            'value': overall_sentiment
                                        }
                                    }
                                ))
                                fig_sentiment.update_layout(height=300)
                                st.plotly_chart(fig_sentiment, use_container_width=True)
                        else:
                            st.warning("No news articles found for this stock.")
                    else:
                        st.error("TextBlob library not available. Please install it with: pip install textblob")

            else:
                st.error(f"No data found for ticker '{ticker}'. Please check the ticker symbol and try again.")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please check your input and try again.")
    else:
        st.info("Please enter a ticker symbol to start.")

if __name__ == "__main__":
    main()
