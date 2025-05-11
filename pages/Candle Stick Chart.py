import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from bokeh.plotting import figure, column
from bokeh.models import WheelZoomTool, PanTool, BoxZoomTool, ResetTool
import talib
import numpy as np

st.set_page_config(layout="wide", page_title="S&P 500 Candlestick App")

# Define colors for multi-stock comparison
COLORS = ["blue", "red", "green", "orange", "purple", "cyan", "magenta", "yellow"]

# Fetch S&P 500 companies
@st.cache_data
def get_sp500_components():
    try:
        df = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        tickers = df["Symbol"].to_list()
        tickers_companies_dict = dict(zip(df["Symbol"], df["Security"]))
        return tickers, tickers_companies_dict
    except Exception as e:
        st.error(f"Error fetching S&P 500 components: {e}")
        return [], {}

# Load data from Yahoo Finance
@st.cache_data
def load_data_yfinance(symbol, start_date, end_date, interval="1d"):
    try:
        df = yf.download(symbol, start=start_date, end=end_date, interval=interval, progress=False)
        if df.empty:
            st.error(f"No data found for {symbol}")
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df = df.dropna()
        df.index.name = 'Date'
        df.index = pd.to_datetime(df.index, utc=True)  # Ensure UTC timezone
        return df
    except Exception as e:
        st.error(f"Error fetching data from Yahoo Finance: {e}")
        return None

# Fetch financial metrics
@st.cache_data
def get_financial_metrics(symbol):
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return {
            "P/E Ratio": info.get("trailingPE", "N/A"),
            "Market Cap": info.get("marketCap", "N/A"),
            "Dividend Yield": info.get("dividendYield", "N/A")
        }
    except Exception as e:
        st.error(f"Error fetching financial metrics for {symbol}: {e}")
        return {"P/E Ratio": "N/A", "Market Cap": "N/A", "Dividend Yield": "N/A"}

# Fetch stock news
@st.cache_data
def get_stock_news(symbol):
    try:
        ticker = yf.Ticker(symbol)
        news = ticker.news[:5]
        return [{"title": item["title"], "link": item.get("link", "#")} for item in news]
    except Exception as e:
        st.error(f"Error fetching news for {symbol}: {e}")
        return []

# Process data with technical indicators and candlestick patterns
@st.cache_data
def process_data(df, symbol):
    try:
        df = df.copy()
        df["BarColor"] = np.where(df["Open"] > df["Close"], "red", "green")
        df["Date_str"] = df.index.strftime("%Y-%m-%d %H:%M:%S")
        
        if not isinstance(df["Close"], pd.Series):
            st.error("Close column is not a valid Series. Check data structure.")
            return None, None
        
        close_array = df["Close"].to_numpy()
        if close_array.ndim != 1:
            st.error(f"Close array has incorrect dimensions: {close_array.ndim}D instead of 1D")
            return None, None

        df["SMA"] = talib.SMA(close_array, timeperiod=3)
        df["MA"] = talib.MA(close_array, timeperiod=3)
        df["EMA"] = talib.EMA(close_array, timeperiod=3)
        df["WMA"] = talib.WMA(close_array, timeperiod=3)
        df["RSI"] = talib.RSI(close_array, timeperiod=3)
        df["MOM"] = talib.MOM(close_array, timeperiod=3)
        df["DEMA"] = talib.DEMA(close_array, timeperiod=3)
        df["TEMA"] = talib.TEMA(close_array, timeperiod=3)

        patterns = {
            "Doji": talib.CDLDOJI(df["Open"].to_numpy(), df["High"].to_numpy(), df["Low"].to_numpy(), df["Close"].to_numpy()),
            "Hammer": talib.CDLHAMMER(df["Open"].to_numpy(), df["High"].to_numpy(), df["Low"].to_numpy(), df["Close"].to_numpy()),
            "Bullish Engulfing": talib.CDLENGULFING(df["Open"].to_numpy(), df["High"].to_numpy(), df["Low"].to_numpy(), df["Close"].to_numpy())
        }
        pattern_df = pd.DataFrame(patterns, index=df.index)
        pattern_df = pattern_df[pattern_df != 0].dropna(how='all').reset_index()
        pattern_df["Symbol"] = symbol
        pattern_df["Date"] = pattern_df["Date"].dt.strftime("%Y-%m-%d %H:%M:%S")

        return df.reset_index(), pattern_df
    except Exception as e:
        st.error(f"Error processing data: {e}")
        return None, None

# Calculate performance metrics
@st.cache_data
def calculate_performance_metrics(df):
    try:
        returns = df["Close"].pct_change().dropna()
        cumulative_return = (1 + returns).cumprod().iloc[-1] - 1
        annualized_volatility = returns.std() * np.sqrt(252)
        max_drawdown = (df["Close"].cummax() - df["Close"]).max() / df["Close"].cummax().max()
        return {
            "Cumulative Return": cumulative_return * 100,
            "Annualized Volatility": annualized_volatility * 100,
            "Max Drawdown": max_drawdown * 100
        }
    except Exception as e:
        st.error(f"Error calculating performance metrics: {e}")
        return {"Cumulative Return": "N/A", "Annualized Volatility": "N/A", "Max Drawdown": "N/A"}

# Backtest SMA crossover strategy
@st.cache_data
def backtest_sma_crossover(df):
    try:
        df = df.copy()
        df["SMA_Short"] = talib.SMA(df["Close"].to_numpy(), timeperiod=10)
        df["SMA_Long"] = talib.SMA(df["Close"].to_numpy(), timeperiod=50)
        df["Signal"] = np.where(df["SMA_Short"] > df["SMA_Long"], 1, 0)
        df["Position"] = df["Signal"].diff()
        df["Returns"] = df["Close"].pct_change()
        df["Strategy_Returns"] = df["Returns"] * df["Signal"].shift(1)
        cumulative_strategy_return = (1 + df["Strategy_Returns"].dropna()).cumprod().iloc[-1] - 1
        return cumulative_strategy_return * 100
    except Exception as e:
        st.error(f"Error backtesting strategy: {e}")
        return "N/A"

# Convert dataframe to downloadable CSV
@st.cache_data
def convert_df_to_csv(df):
    df = df.copy()
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return df.to_csv(index=False).encode("utf-8")

# Create candlestick chart with multiple stocks
indicator_colors = {"SMA": "orange", "EMA": "violet", "WMA": "blue", "RSI": "yellow", "MOM": "black", "DEMA": "red", 
                    "MA": "tomato", "TEMA": "dodgerblue"}

def create_chart(dfs, symbols, close_line=False, include_vol=False, indicators=[], price_alerts=None, multi_stock=False):
    tools = [WheelZoomTool(), PanTool(), BoxZoomTool(), ResetTool()]
    candle = figure(x_axis_type="datetime", height=500, tools=tools, tooltips=[("Date", "@Date_str"), ("Open", "@Open"), 
                                                                               ("High", "@High"), ("Low", "@Low"), ("Close", "@Close")])
    
    for idx, (df, symbol) in enumerate(zip(dfs, symbols)):
        if multi_stock:
            df["Normalized_Close"] = df["Close"] / df["Close"].iloc[0]
            color = COLORS[idx % len(COLORS)]
            candle.line("Date", "Normalized_Close", color=color, legend_label=symbol, source=df)
        else:
            candle.segment("Date", "Low", "Date", "High", color="black", line_width=0.5, source=df)
            candle.segment("Date", "Open", "Date", "Close", line_color="BarColor", line_width=2 if len(df)>100 else 6, source=df)
            if close_line:
                candle.line("Date", "Close", color="black", source=df)
            for indicator in indicators:
                candle.line("Date", indicator, color=indicator_colors[indicator], line_width=2, source=df, legend_label=indicator)
    
    if price_alerts and not multi_stock:
        for level, color in price_alerts.items():
            candle.line(y=level, x=[dfs[0].Date.min(), dfs[0].Date.max()], line_color=color, line_dash="dashed")

    candle.xaxis.axis_label = "Date"
    candle.yaxis.axis_label = "Price ($)" if not multi_stock else "Normalized Price"
    candle.legend.click_policy = "hide"

    volume = None
    if include_vol and not multi_stock:
        volume = figure(x_axis_type="datetime", height=150, x_range=candle.x_range)
        volume.segment("Date", 0, "Date", "Volume", line_width=2 if len(dfs[0])>100 else 6, line_color="BarColor", alpha=0.8, source=dfs[0])
        volume.yaxis.axis_label = "Volume"

    return column(children=[candle, volume], sizing_mode="scale_width") if volume else candle

# Dashboard
st.title(":green[Candle]:red[stick] Pattern Technical Analysis :tea: :coffee:")

# Sidebar for controls
st.sidebar.markdown("#### S&P 500 Company Selection")
tickers, tickers_companies_dict = get_sp500_components()
if tickers:
    selected_companies = st.sidebar.multiselect("Select Companies", 
                                              options=tickers, 
                                              default=["AAPL"],
                                              format_func=lambda x: f"{x} - {tickers_companies_dict.get(x, x)}")
else:
    st.error("No S&P 500 companies available.")
    selected_companies = []

st.sidebar.markdown("#### Data Interval")
interval = st.sidebar.selectbox("Select Interval", ["1d", "1h", "30m"], index=0)

st.sidebar.markdown("#### Date Range Selection")
# Adjust max date range based on interval
max_date = datetime(2024, 12, 31)
if interval in ["1h", "30m"]:
    max_date = datetime.now().date()
    default_end = max_date
    default_start = max_date - timedelta(days=7)
else:
    default_end = datetime(2022, 12, 31)
    default_start = datetime(2022, 1, 1)

col1, col2 = st.sidebar.columns(2, gap="medium")
with col1:
    start_dt = st.date_input("Start:", value=default_start, min_value=datetime(2020, 1, 1), max_value=max_date)
with col2:
    end_dt = st.date_input("End:", value=default_end, min_value=datetime(2020, 1, 1), max_value=max_date)

multi_stock = st.sidebar.checkbox("Compare Multiple Stocks", value=False)
close_line = st.sidebar.checkbox("Close Prices", disabled=multi_stock)
volume = st.sidebar.checkbox("Include Volume", disabled=multi_stock)

talib_indicators = ["MA", "EMA", "SMA", "WMA", "RSI", "MOM", "DEMA", "TEMA"]
indicators = st.sidebar.multiselect(label="Technical Indicators", options=talib_indicators, disabled=multi_stock)

# Price alerts
st.sidebar.markdown("#### Price Alerts")
upper_alert = st.sidebar.number_input("Upper Price Alert", min_value=0.0, step=1.0, value=0.0)
lower_alert = st.sidebar.number_input("Lower Price Alert", min_value=0.0, step=1.0, value=0.0)
price_alerts = {}
if upper_alert > 0:
    price_alerts[upper_alert] = "red"
if lower_alert > 0:
    price_alerts[lower_alert] = "blue"

# Load and process data
if selected_companies:
    dfs = []
    pattern_dfs = []
    for company in selected_companies:
        raw_df = load_data_yfinance(company, start_dt, end_dt, interval=interval)
        if raw_df is not None:
            processed_df, pattern_df = process_data(raw_df, company)
            if processed_df is not None:
                # Convert start_dt and end_dt to datetime64[ns, UTC]
                start_dt_utc = pd.to_datetime(start_dt).tz_localize('UTC')
                end_dt_utc = pd.to_datetime(end_dt).tz_localize('UTC')
                sub_df = processed_df[(processed_df['Date'] >= start_dt_utc) & 
                                     (processed_df['Date'] <= end_dt_utc)]
                if not sub_df.empty:
                    dfs.append(sub_df)
                    if not pattern_df.empty:
                        pattern_dfs.append(pattern_df)
                else:
                    st.error(f"No data available for {company} in the selected date range.")
            else:
                st.error(f"Failed to process data for {company}.")
        else:
            st.error(f"Failed to load data for {company}.")

    if dfs:
        st.bokeh_chart(create_chart(dfs, selected_companies, close_line, volume, indicators, price_alerts, multi_stock), use_container_width=True)

        for company, df in zip(selected_companies, dfs):
            with st.expander(f"Metrics for {company}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("##### Financial Metrics")
                    metrics = get_financial_metrics(company)
                    st.metric("P/E Ratio", f"{metrics['P/E Ratio']:.2f}" if isinstance(metrics['P/E Ratio'], (int, float)) else metrics['P/E Ratio'])
                    st.metric("Market Cap", f"${metrics['Market Cap']/1e9:.2f}B" if isinstance(metrics['Market Cap'], (int, float)) else metrics['Market Cap'])
                    st.metric("Dividend Yield", f"{metrics['Dividend Yield']*100:.2f}%" if isinstance(metrics['Dividend Yield'], (int, float)) else metrics['Dividend Yield'])
                with col2:
                    st.markdown("##### Performance Metrics")
                    perf_metrics = calculate_performance_metrics(df)
                    st.metric("Cumulative Return", f"{perf_metrics['Cumulative Return']:.2f}%" if isinstance(perf_metrics['Cumulative Return'], (int, float)) else perf_metrics['Cumulative Return'])
                    st.metric("Annualized Volatility", f"{perf_metrics['Annualized Volatility']:.2f}%" if isinstance(perf_metrics['Annualized Volatility'], (int, float)) else perf_metrics['Annualized Volatility'])
                    st.metric("Max Drawdown", f"{perf_metrics['Max Drawdown']:.2f}%" if isinstance(perf_metrics['Max Drawdown'], (int, float)) else perf_metrics['Max Drawdown'])

                st.markdown("##### SMA Crossover Strategy")
                strategy_return = backtest_sma_crossover(df)
                st.metric("Strategy Cumulative Return", f"{strategy_return:.2f}%" if isinstance(strategy_return, (int, float)) else strategy_return)

                csv = convert_df_to_csv(df)
                st.download_button(
                    label=f"Download {company} Data as CSV",
                    data=csv,
                    file_name=f"{company}_data.csv",
                    mime="text/csv",
                )

        if pattern_dfs:
            st.markdown("#### Detected Candlestick Patterns")
            combined_patterns = pd.concat(pattern_dfs)
            st.dataframe(combined_patterns[["Symbol", "Date", "Doji", "Hammer", "Bullish Engulfing"]])

        for company in selected_companies:
            with st.expander(f"News for {company}"):
                news_items = get_stock_news(company)
                if news_items:
                    for item in news_items:
                        st.markdown(f"- [{item['title']}]({item['link']})")
                else:
                    st.write("No news available.")
else:
    st.info("Please select at least one company to display the chart.")