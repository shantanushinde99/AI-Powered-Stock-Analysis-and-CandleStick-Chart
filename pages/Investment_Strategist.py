import os
import yfinance as yf
import streamlit as st
from agno.agent import Agent
from agno.models.google import Gemini
import plotly.graph_objects as go

# Set environment variable for Google API
os.environ["GOOGLE_API_KEY"] = "" #place your own API key of Gemini

# Function to fetch and compare stock data
def compare_stocks(symbols):
    data = {}
    for symbol in symbols:
        try:
            # Fetch stock data
            stock = yf.Ticker(symbol)
            hist = stock.history(period="6mo")  # Fetch last 6 months' data
            
            if hist.empty:
                print(f"No data found for {symbol}, skipping it.")
                continue  # Skip this ticker if no data found
            
            # Calculate overall % change
            data[symbol] = hist['Close'].pct_change().sum()
        
        except Exception as e:
            print(f"Could not retrieve data for {symbol}. Reason: {str(e)}")
            continue  # Skip this ticker if an error occurs

    return data

# Define the Market Analyst Agent
market_analyst = Agent(
    model=Gemini(id="gemini-1.5-flash"),
    description="Analyzes and compares stock performance over time.",
    instructions=[
        "Retrieve and compare stock performance from Yahoo Finance.",
        "Calculate percentage change over a 6-month period.",
        "Rank stocks based on their relative performance."
    ],
    show_tool_calls=True,
    markdown=True
)

# Function to get market analysis
def get_market_analysis(symbols):
    performance_data = compare_stocks(symbols)
    
    if not performance_data:
        return "No valid stock data found for the given symbols."

    analysis = market_analyst.run(f"Compare these stock performances: {performance_data}")
    return analysis.content

# ------------------------------ ---------------------------- #

def get_company_info(symbol):
    stock = yf.Ticker(symbol)
    return {
        "name": stock.info.get("longName", "N/A"),
        "sector": stock.info.get("sector", "N/A"),
        "market_cap": stock.info.get("marketCap", "N/A"),
        "summary": stock.info.get("longBusinessSummary", "N/A"),
    }

def get_company_news(symbol):
    stock = yf.Ticker(symbol)
    news = stock.news[:5]  # Get latest 5 news articles
    return news

company_researcher = Agent(
    model=Gemini(id="gemini-2.0-flash"),
    description="Fetches company profiles, financials, and latest news.",
    instructions=[
        "Retrieve company information from Yahoo Finance.",
        "Summarize latest company news relevant to investors.",
        "Provide sector, market cap, and business overview."
    ],
    markdown=True
)

def get_company_analysis(symbol):
    info = get_company_info(symbol)
    news = get_company_news(symbol)
    response = company_researcher.run(
        f"Provide an analysis for {info['name']} in the {info['sector']} sector.\n"
        f"Market Cap: {info['market_cap']}\n"
        f"Summary: {info['summary']}\n"
        f"Latest News: {news}"
    )
    return response.content

# ----------------------------- Stock strategist agent --------------------------- #
stock_strategist = Agent(
    model=Gemini(id="gemini-1.5-pro"),
    description="Provides investment insights and recommends top stocks.",
    instructions=[
        "Analyze stock performance trends and company fundamentals.",
        "Evaluate risk-reward potential and industry trends.",
        "Provide top stock recommendations for investors."
    ],
    markdown=True
)

def get_stock_recommendations(symbols):
    market_analysis = get_market_analysis(symbols)
    data = {}
    for symbol in stocks_symbols:
        data[symbol] = get_company_analysis(symbol)
    recommendations = stock_strategist.run(
        f"Based on the market analysis: {market_analysis}, and company news {data}"
        f"which stocks would you recommend for investment?"
    )
    return recommendations.content

# -------------------------------- Team Lead agent --------------------------------- #
team_lead = Agent(
    model=Gemini(id="gemini-2.0-flash"),
    description="Aggregates stock analysis, company research, and investment strategy.",
    instructions=[
        "Compile stock performance, company analysis, and recommendations.",
        "Ensure all insights are structured in an investor-friendly report.",
        "Rank the top stocks based on combined analysis."
    ],
    markdown=True
)

def get_final_investment_report(symbols):
    market_analysis = get_market_analysis(symbols)
    company_analyses = [get_company_analysis(symbol) for symbol in symbols]
    stock_recommendations = get_stock_recommendations(symbols)

    final_report = team_lead.run(
    f"""
Market Analysis:
{market_analysis}

Company Analyses:
{company_analyses}

Stock Recommendations:
{stock_recommendations}

You are to generate a professional, investor-friendly report. Use the following format for each stock:

## [Company Name] ([Ticker]) Analysis

**Sector:** [Sector]  
**Market Cap:** [Market Cap]

### Business Overview:
[Business overview in 2-3 lines.]

### Fundamentals:
* **P/E Ratio:** 
* **Revenue Growth (YoY):** 
* **Debt-to-Equity Ratio:** 
* **Gross Margin:** 

### Latest News Summary:
* [Summarize the 3–5 most relevant news headlines.]

### SWOT Analysis:
* **Strengths:** 
* **Weaknesses:** 
* **Opportunities:** 
* **Threats:** 

### Investment Recommendation: **[BUY | HOLD | SELL]**

**Rationale:**  
[Give a data-driven explanation for the recommendation.]

**Key Monitoring Points:**
- [Point 1]
- [Point 2]
- [Point 3]

---

### Final Ranked List (Buy Recommendation - Ascending Order)
Rank all the stocks from least to most recommended for buying.

---

**Disclaimer:** *This analysis is generated by AI for educational purposes and should not be considered financial advice. Investors should perform their own research or consult a licensed advisor before making any decisions.*
"""
)

    return final_report.content


# Streamlit page configuration
st.set_page_config(page_title="AI Investment Strategist", page_icon="📈", layout="wide")

# Title and header
st.markdown("""
    <h1 style="text-align: center; color: #4CAF50;">📈 AI Investment Strategist</h1>
    <h3 style="text-align: center; color: #6c757d;">Generate personalized investment reports with the latest market insights.</h3>
""", unsafe_allow_html=True)

# Sidebar Styling
st.sidebar.markdown("""
    <h2 style="color: #343a40;">Configuration</h2>
    <p style="color: #6c757d;">Enter the stock symbols you want to analyze. The AI will provide detailed insights, performance reports, and top recommendations.</p>
""", unsafe_allow_html=True)

# Stock symbols input
input_symbols = st.sidebar.text_input("Enter Stock Symbols (separated by commas)", "AAPL, TSLA, GOOG")
api_key = st.sidebar.text_input("Enter your API Key (optional)", type="password")

# Parse the stock symbols input
stocks_symbols = [symbol.strip() for symbol in input_symbols.split(",")]

# Generate Investment Report button
if st.sidebar.button("Generate Investment Report"):
    if not stocks_symbols:
        st.sidebar.warning("Please enter at least one stock symbol.")
    elif not api_key:
        st.sidebar.warning("Please enter your API Key.")
    else:
        # Generate the final report
        report = get_final_investment_report(stocks_symbols)
        
        # Display the report
        st.subheader("Investment Report")
        st.markdown(report)

        st.info("This report provides detailed insights, including market performance, company analysis, and investment recommendations.")

        # Interactive Stock Performance Chart
        st.markdown("### 📊 Stock Performance (6-Months)")
        stock_data = yf.download(stocks_symbols, period="6mo")['Close']

        fig = go.Figure()
        for symbol in stocks_symbols:
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data[symbol], mode='lines', name=symbol))

        fig.update_layout(title="Stock Performance Over the Last 6 Months",
                          xaxis_title="Date",
                          yaxis_title="Price (in USD)",
                          template="plotly_dark")
        st.plotly_chart(fig)
