import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import plotly.subplots as sp
import streamlit as st
import traceback
from datetime import date, timedelta

# Initialize session state
if 'entry_rsi' not in st.session_state:
    st.session_state.entry_rsi = 30
if 'exit_rsi' not in st.session_state:
    st.session_state.exit_rsi = 70
if 'window' not in st.session_state:
    st.session_state.window = 14
if 'interval' not in st.session_state:
    st.session_state.interval = '1d'
if 'start_date' not in st.session_state:
    st.session_state.start_date = date(2022, 1, 1)
if 'end_date' not in st.session_state:
    st.session_state.end_date = date.today()

# Function to calculate RSI
def calculate_rsi(data, window=14):
    try:
        delta = data['Close'].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    except Exception as e:
        st.error("Error in calculate_rsi:")
        st.error(traceback.format_exc())

# Function to calculate strategy returns
def calculate_strategy_returns(data, entry_rsi, exit_rsi, window):
    try:
        data['RSI'] = calculate_rsi(data, window)
        data['Position'] = 0  # Initial position is neutral (0)
        
        position = 0
        for i in range(1, len(data)):
            if data['RSI'].iloc[i] < entry_rsi and position == 0:
                position = 1  # Enter long position
            elif data['RSI'].iloc[i] > exit_rsi and position == 1:
                position = 0  # Exit long position
            data.at[data.index[i], 'Position'] = position
        
        data['Daily Return'] = data['Close'].pct_change()
        data['Strategy Return'] = data['Daily Return'] * data['Position'].shift(1)
        data['Cumulative Strategy Return'] = (1 + data['Strategy Return']).cumprod() - 1
        
        return data
    except Exception as e:
        st.error("Error in calculate_strategy_returns:")
        st.error(traceback.format_exc())

# Function to calculate returns only during the testing period
def calculate_testing_strategy_returns(train_data, test_data, entry_rsi, exit_rsi, window):
    try:
        combined_data = pd.concat([train_data, test_data])
        combined_data['RSI'] = calculate_rsi(combined_data, window)
        combined_data['Position'] = 0  # Initial position is neutral (0)
        
        position = 0
        for i in range(len(train_data), len(combined_data)):
            if combined_data['RSI'].iloc[i] < entry_rsi and position == 0:
                position = 1  # Enter long position
            elif combined_data['RSI'].iloc[i] > exit_rsi and position == 1:
                position = 0  # Exit long position
            combined_data.at[combined_data.index[i], 'Position'] = position
        
        combined_data['Daily Return'] = combined_data['Close'].pct_change()
        combined_data['Strategy Return'] = combined_data['Daily Return'] * combined_data['Position'].shift(1)
        combined_data['Cumulative Strategy Return'] = (1 + combined_data['Strategy Return']).cumprod() - 1
        combined_data['Cumulative Buy Hold Return'] = (1 + combined_data['Daily Return'].iloc[len(train_data):]).cumprod() - 1
        
        return combined_data
    except Exception as e:
        st.error("Error in calculate_testing_strategy_returns:")
        st.error(traceback.format_exc())

# Function to plot the stock data and RSI strategy
def plot_stock_and_rsi_strategy(data, ticker, entry_rsi, exit_rsi, window, split_index):
    try:
        train_data = data.iloc[:split_index]
        test_data = data.iloc[split_index:]
        
        # Calculate the final cumulative returns
        final_cumulative_strategy_return = test_data['Cumulative Strategy Return'].iloc[-1] * 100
        final_cumulative_buy_hold_return = test_data['Cumulative Buy Hold Return'].iloc[-1] * 100

        fig = sp.make_subplots(rows=3, cols=1, shared_xaxes=True,
                               subplot_titles=(f"{ticker} Stock Price",
                                               f"{ticker} RSI (Window={window})",
                                               f"{ticker} Cumulative Returns"))

        # Plot closing prices
        fig.add_trace(go.Scatter(x=train_data.index, y=train_data['Close'], mode='lines', name='Close Price (Train)', line=dict(color='blue')),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=test_data.index, y=test_data['Close'], mode='lines', name='Close Price (Test)', line=dict(color='red')),
                      row=1, col=1)
        
        # Plot RSI
        fig.add_trace(go.Scatter(x=train_data.index, y=train_data['RSI'], mode='lines', name='RSI (Train)', line=dict(color='orange')),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=test_data.index, y=test_data['RSI'], mode='lines', name='RSI (Test)', line=dict(color='purple')),
                      row=2, col=1)
        fig.add_shape(type="line", x0=data.index[0], x1=data.index[-1], y0=entry_rsi, y1=entry_rsi,
                      line=dict(color='green', dash='dash'), row=2, col=1)
        fig.add_shape(type="line", x0=data.index[0], x1=data.index[-1], y0=exit_rsi, y1=exit_rsi,
                      line=dict(color='red', dash='dash'), row=2, col=1)
        
        # Plot cumulative strategy return
        fig.add_trace(go.Scatter(x=test_data.index, y=test_data['Cumulative Strategy Return'] * 100, mode='lines', name='Cumulative Strategy Return (Test)', line=dict(color='green')),
                      row=3, col=1)
        # Plot cumulative buy-and-hold return
        fig.add_trace(go.Scatter(x=test_data.index, y=test_data['Cumulative Buy Hold Return'] * 100, mode='lines', name='Cumulative Buy and Hold Return (Test)', line=dict(color='blue')),
                      row=3, col=1)
        
        # Add annotation for the final cumulative returns
        fig.add_annotation(x=test_data.index[-1], y=final_cumulative_strategy_return,
                           text=f"Final Strategy Return (Test): {final_cumulative_strategy_return:.2f}%",
                           showarrow=True, arrowhead=1, row=3, col=1)
        fig.add_annotation(x=test_data.index[-1], y=final_cumulative_buy_hold_return,
                           text=f"Final Buy & Hold Return (Test): {final_cumulative_buy_hold_return:.2f}%",
                           showarrow=True, arrowhead=1, row=3, col=1)
        fig.add_vline(x=train_data.index[-1], line=dict(color='black', dash='dash'), row=3, col=1)

        # Update layout
        fig.update_layout(height=900, width=900, title_text=f"{ticker} RSI Trading Strategy Analysis", showlegend=False)
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Close Price", row=1, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1)
        fig.update_yaxes(title_text="Cumulative % Return", row=3, col=1)
        
        st.plotly_chart(fig)
    except Exception as e:
        st.error("Error in plot_stock_and_rsi_strategy:")
        st.error(traceback.format_exc())

# Function to find the best RSI combination using train-test split with a progress bar
def optimize_rsi(ticker, start_date, end_date, interval):
    try:
        data = yf.download(ticker, start=start_date, end=end_date + timedelta(days=1), interval=interval)  # Adjust end date

        if data.empty:
            st.error("No data fetched. Please check the ticker symbol or date range.")
            return None, None, None, None
        
        # Split data into training and testing sets (e.g., first 70% for training, last 30% for testing)
        split_index = int(len(data) * 0.5)
        train_data = data.iloc[:split_index]
        test_data = data.iloc[split_index:]

        best_entry_rsi = None
        best_exit_rsi = None
        best_window = None
        best_return = float('-inf')
        
        param_combinations = [(window, entry_rsi, exit_rsi) 
                              for window in range(10, 31, 2) 
                              for entry_rsi in range(0, 51, 2) 
                              for exit_rsi in range(50, 101, 2)]
        
        progress_bar = st.progress(0)
        total_combinations = len(param_combinations)
        progress = 0
        
        for window, entry_rsi, exit_rsi in param_combinations:
            temp_train_data = calculate_strategy_returns(train_data.copy(), entry_rsi, exit_rsi, window)
            temp_test_data = calculate_testing_strategy_returns(train_data.copy(), test_data.copy(), entry_rsi, exit_rsi, window)
            final_return = temp_test_data['Cumulative Strategy Return'].iloc[-1]
            if final_return > best_return:
                best_return = final_return
                best_entry_rsi = entry_rsi
                best_exit_rsi = exit_rsi
                best_window = window
            progress += 1
            progress_bar.progress(progress / total_combinations)
        
        return best_entry_rsi, best_exit_rsi, best_window, best_return
    except Exception as e:
        st.error("Error in optimize_rsi:")
        st.error(traceback.format_exc())

# Streamlit app
st.title("RSI Trading Strategy Optimization")

tickers = st.text_input('Tickers (comma separated)', 'AAPL,MSFT,GOOG')
entry_rsi = st.slider('Entry RSI', min_value=0, max_value=50, value=st.session_state.entry_rsi, step=1)
exit_rsi = st.slider('Exit RSI', min_value=50, max_value=100, value=st.session_state.exit_rsi, step=1)
window = st.slider('RSI Window', min_value=10, max_value=30, value=st.session_state.window, step=1)
interval = st.selectbox('Interval', ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'], index=7)
start_date = st.date_input('Start Date', value=st.session_state.start_date)
end_date = st.date_input('End Date', value=st.session_state.end_date)

# Adjust displayed dates by subtracting one day
adjusted_start_date = start_date - timedelta(days=1)
adjusted_end_date = end_date - timedelta(days=1)

# Display a warning if the selected interval is restricted
if interval in ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h'] and (end_date - start_date).days > 60:
    st.warning("Intraday intervals (interval <1d) are only available for the last 60 days.")
if interval == '1m' and (end_date - start_date).days > 7:
    st.warning("1-minute interval data is only available for the last 7 days.")

optimize_button = st.button('Optimize RSI')
show_button = st.button('Show RSI Strategy Graph')

if show_button:
    tickers_list = [ticker.strip().upper() for ticker in tickers.split(',')]
    for ticker in tickers_list:
        if not ticker:
            continue
        try:
            data = yf.download(ticker, start=adjusted_start_date, end=adjusted_end_date + timedelta(days=1), interval=interval)  # Adjust end date
            if data.empty:
                st.error(f"No data fetched for {ticker}. Please check the ticker symbol or date range.")
                continue
            split_index = int(len(data) * 0.5)
            data = calculate_testing_strategy_returns(data.iloc[:split_index], data.iloc[split_index:], entry_rsi, exit_rsi, window)
            plot_stock_and_rsi_strategy(data, ticker, entry_rsi, exit_rsi, window, split_index)
        except Exception as e:
            st.error(f"Error processing {ticker}:")
            st.error(traceback.format_exc())

if optimize_button:
    tickers_list = [ticker.strip().upper() for ticker in tickers.split(',')]
    for ticker in tickers_list:
        if not ticker:
            continue
        try:
            best_entry_rsi, best_exit_rsi, best_window, best_return = optimize_rsi(ticker, adjusted_start_date, adjusted_end_date, interval)
            if best_entry_rsi is not None and best_exit_rsi is not None and best_window is not None:
                st.success(f"{ticker} - Optimal Entry RSI: {best_entry_rsi}, Optimal Exit RSI: {best_exit_rsi}, Optimal Window: {best_window}, Best Return: {best_return * 100:.2f}%")
                entry_rsi = best_entry_rsi
                exit_rsi = best_exit_rsi
                window = best_window
                data = yf.download(ticker, start=adjusted_start_date, end=adjusted_end_date + timedelta(days=1), interval=interval)  # Adjust end date
                if data.empty:
                    st.error(f"No data fetched for {ticker}. Please check the ticker symbol or date range.")
                    continue
                split_index = int(len(data) * 0.5)
                data = calculate_testing_strategy_returns(data.iloc[:split_index], data.iloc[split_index:], best_entry_rsi, best_exit_rsi, best_window)
                plot_stock_and_rsi_strategy(data, ticker, best_entry_rsi, best_exit_rsi, best_window, split_index)
        except Exception as e:
            st.error(f"Error optimizing {ticker}:")
            st.error(traceback.format_exc())
