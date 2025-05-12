# 📊 AI Investment Dashboard 🚀

Welcome to the **AI Investment Dashboard**, a powerful Streamlit-based platform for investors and traders! This project offers two core modules:

- 📈 **Investment Plan App**: Generate personalized investment plans with AI-driven recommendations (BUY, HOLD, SELL) based on market performance, company fundamentals, and financial news.
- 📉 **Technical Analysis App**: Analyze stock trends with candlestick charts, technical indicators (e.g., SMA, RSI), intraday data, price alerts, and multi-stock comparisons.

Built with Python, Streamlit, and yfinance, this dashboard is perfect for long-term investors and short-term traders alike. 🌟
This is for Educational Purposes Only - Not Recommended for Professsional Uses

## 🎥 Demo Video

Watch a quick demo of the AI Investment Dashboard in action! 📽️

![Demo](https://github.com/shantanushinde99/AI-Powered-Stock-Analysis-and-CandleStick-Chart/blob/main/images/Demo1.gif?raw=true)

Full Video is there with Latest Features as well [full video](https://github.com/shantanushinde99/AI-Powered-Stock-Analysis-and-CandleStick-Chart/blob/main/Full%20Video_2.mp4)

## 🖼️ Screenshots

Here’s a glimpse of the dashboard’s sleek interface! 🖥️

| **Dashboard Landing Page** | **Technical Analysis App** |
|----------------------------|----------------------------|
| ![Dashboard](images/Screenshot(107).png) | ![Technical Analysis](images/Screenshot(108).png) |

## ✨ Features

### Investment Plan App 📈
- **AI Recommendations**: Get BUY, HOLD, SELL signals based on market data and sentiment analysis.
- **Comprehensive Reports**: Generate detailed investment plans with company overviews and ranked opportunities.
- **News Integration**: Stay informed with real-time financial news.

### Technical Analysis App 📉
- **Candlestick Charts**: Visualize stock price movements with interactive Bokeh charts.
- **Technical Indicators**: Analyze trends with SMA, EMA, RSI, MACD, and more.
- **Intraday Data**: Support for 1-hour and 30-minute intervals for precise trading.
- **Price Alerts**: Set upper/lower price thresholds with visual alerts on charts.
- **Multi-Stock Comparison**: Compare normalized price trends across multiple stocks.

## 🛠️ Installation

Get started in just a few steps! 🔧 Use Python 3.11 

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/ai-investment-dashboard.git
   cd ai-investment-dashboard
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install streamlit pandas yfinance bokeh numpy TA-Lib
   ```

   *Note*: For `TA-Lib`, you may need to install the binary first:
   ```bash
   pip install TA-Lib
   ```
   If issues arise, follow the [TA-Lib installation guide](https://github.com/TA-Lib/ta-lib-python).

## 🚀 Usage

Run the dashboard with a single command! 🏃‍♂️

```bash
streamlit run Home Page.py
```

- Open your browser at `http://localhost:8501`.
- Use the sidebar to select between the **Investment Plan App** and **Technical Analysis App**.
- For the **Technical Analysis App**:
  - Choose S&P 500 companies.
  - Select a date range and data interval (`1d`, `1h`, `30m`).
  - Add technical indicators and price alerts for analysis.


## 🌟 What's New

Recent updates to make your experience even better! 🎉

- **Intraday Data Support**: Analyze stocks with 1-hour or 30-minute intervals for precise trading.
- **Price Alerts**: Visualize key price levels with dashed lines on candlestick charts.
- **Multi-Stock Comparison**: Compare multiple stocks with normalized price trends.
- **Robust Datetime Handling**: Fixed datetime issues for accurate date range filtering.
- **Enhanced AI Recommendations** (Investment Plan App): Improved BUY/SELL signals with sentiment analysis.


## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details. 📄

## 📬 Contact

Have questions or feedback? Reach out! ✉️

- **GitHub**: [Shantanu Shinde](https://github.com/shantanushinde99)
- **Email**: shantanushinde233@gmail.com


