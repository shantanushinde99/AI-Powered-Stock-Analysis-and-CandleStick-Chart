import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, date

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
def load_data_yfinance(symbol, start_date, end_date):
    try:
        df = yf.download(symbol, start=start_date, end=end_date, progress=False)
        if df.empty:
            st.error(f"No data found for {symbol}")
            return None
        # Select required columns and ensure proper formatting
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df = df.dropna()  # Drop rows with missing values
        df.index.name = 'Date'  # Ensure index is named 'Date'
        return df
    except Exception as e:
        st.error(f"Error fetching data from Yahoo Finance: {e}")
        return None

# Convert dataframe to downloadable CSV
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv().encode("utf-8")

# Calculate technical indicators
def calculate_indicators(df, sma_periods, bb_periods, bb_std, rsi_periods):
    df = df.copy()
    # SMA
    if sma_periods > 0:
        df['SMA'] = df['Close'].rolling(window=sma_periods).mean()
    # Bollinger Bands
    if bb_periods > 0:
        df['BB_Middle'] = df['Close'].rolling(window=bb_periods).mean()
        df['BB_Std'] = df['Close'].rolling(window=bb_periods).std()
        df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * bb_std)
        df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * bb_std)
    # RSI
    if rsi_periods > 0:
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_periods).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
    return df

# Sidebar UI
st.sidebar.header("Stock Parameters")
available_tickers, tickers_companies_dict = get_sp500_components()

if not available_tickers:
    st.error("No tickers available. Please check your internet connection or try again later.")
else:
    ticker = st.sidebar.selectbox("Ticker", available_tickers, format_func=tickers_companies_dict.get)
    start_date = st.sidebar.date_input("Start date", date(2019, 1, 1))
    end_date = st.sidebar.date_input("End date", date.today())

    if start_date > end_date:
        st.sidebar.error("The end date must fall after the start date")

    # Technical indicator options
    st.sidebar.header("Technical Analysis Parameters")

    volume_flag = st.sidebar.checkbox("Add volume")

    exp_sma = st.sidebar.expander("SMA")
    sma_flag = exp_sma.checkbox("Add SMA")
    sma_periods = exp_sma.number_input("SMA Periods", 1, 50, 20)

    exp_bb = st.sidebar.expander("Bollinger Bands")
    bb_flag = exp_bb.checkbox("Add Bollinger Bands")
    bb_periods = exp_bb.number_input("BB Periods", 1, 50, 20)
    bb_std = exp_bb.number_input("Standard Deviations", 1, 4, 2)

    exp_rsi = st.sidebar.expander("RSI")
    rsi_flag = exp_rsi.checkbox("Add RSI")
    rsi_periods = exp_rsi.number_input("RSI Periods", 1, 50, 20)
    rsi_upper = exp_rsi.number_input("RSI Upper", 50, 90, 70)
    rsi_lower = exp_rsi.number_input("RSI Lower", 10, 50, 30)

    # Main App Title
    st.title("📊 Technical Analysis Dashboard (Powered by Yahoo Finance)")
    st.markdown("""
    Select a stock from the S&P 500 and apply technical indicators for visual analysis.
    """)

    # Load data
    df = load_data_yfinance(ticker, start_date, end_date)

    if df is not None and not df.empty:
        # Preview section
        data_exp = st.expander("📄 Preview Data")
        available_cols = df.columns.tolist()
        columns_to_show = data_exp.multiselect("Columns to display", available_cols, default=available_cols)
        data_exp.dataframe(df[columns_to_show])

        csv_file = convert_df_to_csv(df[columns_to_show])
        data_exp.download_button(
            label="📥 Download CSV",
            data=csv_file,
            file_name=f"{ticker}_yfinance_data.csv",
            mime="text/csv",
        )

        # Calculate indicators
        df = calculate_indicators(df, sma_periods if sma_flag else 0, bb_periods if bb_flag else 0, bb_std, rsi_periods if rsi_flag else 0)

        # Create subplots
        try:
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                                subplot_titles=[f"{tickers_companies_dict[ticker]} Stock Price", "Volume", "RSI"],
                                row_heights=[0.5, 0.2, 0.3])

            # Candlestick chart
            fig.add_trace(go.Candlestick(x=df.index,
                                        open=df['Open'],
                                        high=df['High'],
                                        low=df['Low'],
                                        close=df['Close'],
                                        name="OHLC"), row=1, col=1)

            # SMA
            if sma_flag and 'SMA' in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df['SMA'], name="SMA", line=dict(color='orange')), row=1, col=1)

            # Bollinger Bands
            if bb_flag and all(col in df.columns for col in ['BB_Upper', 'BB_Lower', 'BB_Middle']):
                fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name="BB Upper", line=dict(color='green', dash='dash')), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name="BB Lower", line=dict(color='red', dash='dash')), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['BB_Middle'], name="BB Middle", line=dict(color='blue')), row=1, col=1)

            # Volume
            if volume_flag:
                fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name="Volume", marker_color='black'), row=2, col=1)

            # RSI
            if rsi_flag and 'RSI' in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI", line=dict(color='violet')), row=3, col=1)
                fig.add_hline(y=rsi_upper, line_dash="dash", line_color="red", row=3, col=1)
                fig.add_hline(y=rsi_lower, line_dash="dash", line_color="green", row=3, col=1)

            # Update layout
            fig.update_layout(
                title=f"{tickers_companies_dict[ticker]} Technical Analysis",
                yaxis_title="Price",
                xaxis_rangeslider_visible=False,
                showlegend=True,
                height=800
            )

            # Hide volume/RSI subplots if not selected
            if not volume_flag:
                fig.update_yaxes(visible=False, row=2, col=1)
            if not rsi_flag:
                fig.update_yaxes(visible=False, row=3, col=1)

            # Display plot
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error plotting data: {e}")
            st.markdown("""
            **Note**: An error occurred while plotting. Please ensure all parameters are valid and try again.
            If the issue persists, check your `plotly` installation:
            ```bash
            pip install --upgrade plotly
            ```
            """)