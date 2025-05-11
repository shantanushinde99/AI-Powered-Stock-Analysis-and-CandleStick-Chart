import streamlit as st

# Set page config
st.set_page_config(
    page_title="📊 AI Investment Dashboard",
    page_icon="📈",
    layout="wide"
)

# Title and welcome message
st.title("📊 AI Investment Dashboard")
st.markdown("""
Welcome to the AI Investment Platform.
""")

st.markdown("""
Use the sidebar to switch between:

- 📈 **Investment Plan App**  
  Detailed investment plans based on market performance, company fundamentals, and recent financial news. It generates personalized reports with company overviews, stock analysis, recommendations (BUY, HOLD, SELL), and a ranked list of opportunities to support long-term investment strategies.

- 📉 **Technical Analysis App**  
  Dive deep into market trends with this tool. It analyzes stock charts using popular technical indicators like moving averages, RSI, MACD, and more. Ideal for traders who rely on technical signals to make short- to medium-term trading decisions.
""")

st.markdown("---")

st.info("Use the **left sidebar** to select a module.")
