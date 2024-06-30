import yfinance as yf
import pandas as pd
import streamlit as st
import plotly.graph_objs as go
import plotly.subplots as sp
from datetime import timedelta
import traceback

def calculate_testing_strategy_returns(train_data, test_data, entry_rsi, exit_rsi, window):
    # Function to calculate the strategy returns. 
    # This is a placeholder and should be replaced with the actual implementation.
    test_data['Cumulative Strategy Return'] = 1  # Placeholder calculation
    test_data['Cumulative Buy Hold Return'] = 1  # Placeholder calculation
    return pd.concat([train_data, test_data])

def plot_stock_and_rsi_strategy(data, ticker, entry_rsi, exit_rsi, window, split_index):
    try:
        train_data = data.iloc[:split_index]
        test_data = data.iloc[split_index:]

        final_cumulative_strategy_return = test_data['Cumulative Strategy Return'].iloc[-1] * 100
        final_cumulative_buy_hold_return = test_data['Cumulative Buy Hold Return'].iloc[-1] * 100

        fig = sp.make_subplots(rows=3, cols=1, shared_xaxes=True,
                               subplot_titles=(f"{ticker} Stock Price",
                                               f"{ticker} RSI (Window={window})",
                                               f"{ticker} Cumulative Returns"))

        fig.add_trace(go.Scatter(x=train_data.index, y=train_data['Close'], mode='lines', name='Close Price (Train)', line=dict(color='blue')),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=test_data.index, y=test_data['Close'], mode='lines', name='Close Price (Test)', line=dict(color='red')),
                      row=1, col=1)

        fig.add_trace(go.Scatter(x=train_data.index, y=train_data['RSI'], mode='lines', name='RSI (Train)', line=dict(color='orange')),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=test_data.index, y=test_data['RSI'], mode='lines', name='RSI (Test)', line=dict(color='purple')),
                      row=2, col=1)
        fig.add_shape(type="line", x0=data.index[0], x1=data.index[-1], y0=entry_rsi, y1=entry_rsi,
                      line=dict(color='green', dash='dash'), row=2, col=1)
        fig.add_shape(type="line", x0=data.index[0], x1=data.index[-1], y0=exit_rsi, y1=exit_rsi,
                      line=dict(color='red', dash='dash'), row=2, col=1)

        fig.add_trace(go.Scatter(x=test_data.index, y=test_data['Cumulative Strategy Return'] * 100, mode='lines', name='Cumulative Strategy Return (Test)', line=dict(color='green')),
                      row=3, col=1)
        fig.add_trace(go.Scatter(x=test_data.index, y=test_data['Cumulative Buy Hold Return'] * 100, mode='lines', name='Cumulative Buy and Hold Return (Test)', line=dict(color='blue')),
                      row=3, col=1)

        fig.add_annotation(x=test_data.index[-1], y=final_cumulative_strategy_return,
                           text=f"Final Strategy Return (Test): {final_cumulative_strategy_return:.2f}%",
                           showarrow=True, arrowhead=1, row=3, col=1)
        fig.add_annotation(x=test_data.index[-1], y=final_cumulative_buy_hold_return,
                           text=f"Final Buy & Hold Return (Test): {final_cumulative_buy_hold_return:.2f}%",
                           showarrow=True, arrowhead=1, row=3, col=1)
        fig.add_vline(x=train_data.index[-1], line=dict(color='black', dash='dash'), row=3, col=1)

        fig.update_layout(height=900, width=900, title_text=f"{ticker} RSI Trading Strategy Analysis", showlegend=False)
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Close Price", row=1, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1)
        fig.update_yaxes(title_text="Cumulative % Return", row=3, col=1)

        st.plotly_chart(fig)
    except Exception as e:
        st.error("Error in plot_stock_and_rsi_strategy:")
        st.error(traceback.format_exc())

# Adjusted Streamlit app
st.title("RSI Trading Strategy Backtest")

# Define the UI elements
tickers = st.text_input("Enter tickers separated by commas (e.g., AAPL,MSFT,GOOGL):")
start_date = st.date_input("Start Date")
end_date = st.date_input("End Date")
interval = st.selectbox("Interval", ['1d', '1wk', '1mo'])
entry_rsi = st.slider("Entry RSI", min_value=0, max_value=100, value=30)
exit_rsi = st.slider("Exit RSI", min_value=0, max_value=100, value=70)
window = st.slider("RSI Window", min_value=1, max_value=50, value=14)
show_button = st.button("Show Analysis")

if show_button:
    tickers_list = [ticker.strip().upper() for ticker in tickers.split(',')]
    for ticker in tickers_list:
        if not ticker:
            continue
        try:
            data = yf.download(ticker, start=start_date, end=end_date + timedelta(days=1), interval=interval)  # Adjust end date
            if data.empty:
                st.error(f"No data fetched for {ticker}. Please check the ticker symbol or date range.")
                continue
            split_index = int(len(data) * 0.7)
            data = calculate_testing_strategy_returns(data.iloc[:split_index], data.iloc[split_index:], entry_rsi, exit_rsi, window)
            plot_stock_and_rsi_strategy(data, ticker, entry_rsi, exit_rsi, window, split_index)
        except Exception as e:
            st.error(f"Error processing {ticker}:")
            st.error(traceback.format_exc())
