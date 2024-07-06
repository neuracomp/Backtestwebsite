import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import plotly.subplots as sp
import streamlit as st
import traceback
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor

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
if 'days_range' not in st.session_state:
    st.session_state.days_range = 30
if 'train_split' not in st.session_state:
    st.session_state.train_split = 70

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
        return pd.Series([])

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
        data['Cumulative Buy Hold Return'] = (1 + data['Daily Return']).cumprod() - 1
        
        return data
    except Exception as e:
        st.error("Error in calculate_strategy_returns:")
        st.error(traceback.format_exc())
        return pd.DataFrame()

# Function to plot the stock data and RSI strategy
def plot_stock_and_rsi_strategy(data, ticker, entry_rsi, exit_rsi, window):
    try:
        # Calculate the final cumulative returns
        final_cumulative_strategy_return = data['Cumulative Strategy Return'].iloc[-1] * 100
        final_cumulative_buy_hold_return = data['Cumulative Buy Hold Return'].iloc[-1] * 100

        fig = sp.make_subplots(rows=3, cols=1, shared_xaxes=True,
                               subplot_titles=(f"{ticker} Stock Price",
                                               f"{ticker} RSI (Window={window})",
                                               f"{ticker} Cumulative Returns"))

        # Plot closing prices
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price', line=dict(color='blue')),
                      row=1, col=1)
        
        # Plot RSI
        fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], mode='lines', name='RSI', line=dict(color='orange')),
                      row=2, col=1)
        fig.add_shape(type="line", x0=data.index[0], x1=data.index[-1], y0=entry_rsi, y1=entry_rsi,
                      line=dict(color='green', dash='dash'), row=2, col=1)
        fig.add_shape(type="line", x0=data.index[0], x1=data.index[-1], y0=exit_rsi, y1=exit_rsi,
                      line=dict(color='red', dash='dash'), row=2, col=1)
        
        # Plot cumulative strategy return
        fig.add_trace(go.Scatter(x=data.index, y=data['Cumulative Strategy Return'] * 100, mode='lines', name='Cumulative Strategy Return', line=dict(color='green')),
                      row=3, col=1)
        # Plot cumulative buy-and-hold return
        fig.add_trace(go.Scatter(x=data.index, y=data['Cumulative Buy Hold Return'] * 100, mode='lines', name='Cumulative Buy and Hold Return', line=dict(color='blue')),
                      row=3, col=1)
        
        # Add annotation for the final cumulative returns
        fig.add_annotation(x=data.index[-1], y=final_cumulative_strategy_return,
                           text=f"Final Strategy Return: {final_cumulative_strategy_return:.2f}%",
                           showarrow=True, arrowhead=1, row=3, col=1)
        fig.add_annotation(x=data.index[-1], y=final_cumulative_buy_hold_return,
                           text=f"Final Buy & Hold Return: {final_cumulative_buy_hold_return:.2f}%",
                           showarrow=True, arrowhead=1, row=3, col=1)

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

# Function to find the best RSI combination using brute force with a progress bar
def optimize_rsi(ticker, start_date, end_date, interval, train_split):
    try:
        data = yf.download(ticker, start=start_date, end=end_date + timedelta(days=1), interval=interval)  # Adjust end date

        if data.empty:
            st.error("No data fetched. Please check the ticker symbol or date range.")
            return None, None, None, None

        split_index = int(len(data) * train_split / 100)
        train_data = data.iloc[:split_index]
        test_data = data.iloc[split_index:]

        best_entry_rsi = None
        best_exit_rsi = None
        best_window = None
        best_return = float('-inf')
        
        param_combinations = [(window, entry_rsi, exit_rsi) 
                              for window in range(10, 30, 2) 
                              for entry_rsi in range(0, 51, 5) 
                              for exit_rsi in range(50, 101, 5)]
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        total_combinations = len(param_combinations)
        progress = 0
        
        def evaluate_combination(params):
            window, entry_rsi, exit_rsi = params
            temp_data = calculate_strategy_returns(train_data.copy(), entry_rsi, exit_rsi, window)
            final_return = temp_data['Cumulative Strategy Return'].iloc[-1]
            return window, entry_rsi, exit_rsi, final_return
        
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(evaluate_combination, param_combinations))
        
        for idx, (window, entry_rsi, exit_rsi, final_return) in enumerate(results):
            if final_return > best_return:
                best_return = final_return
                best_entry_rsi = entry_rsi
                best_exit_rsi = exit_rsi
                best_window = window
            progress += 1
            progress_bar.progress(progress / total_combinations)
            status_text.text(f"Evaluating combination {idx + 1}/{total_combinations}: Window={window}, Entry RSI={entry_rsi}, Exit RSI={exit_rsi}")
        
        return best_entry_rsi, best_exit_rsi, best_window, best_return
    except Exception as e:
        st.error("Error in optimize_rsi:")
        st.error(traceback.format_exc())

# Streamlit app
st.title("RSI Trading Strategy Optimization")

tickers = st.text_input('Tickers (comma separated)', 'SPY')
entry_rsi = st.slider('Entry RSI', min_value=0, max_value=50, value=st.session_state.entry_rsi, step=1, help='The RSI value below which the strategy will enter a long position.')
exit_rsi = st.slider('Exit RSI', min_value=50, max_value=100, value=st.session_state.exit_rsi, step=1, help='The RSI value above which the strategy will exit a long position.')
window = st.slider('RSI Window', min_value=10, max_value=30, value=st.session_state.window, step=1, help='The window size for calculating RSI.')
interval = st.selectbox('Interval', ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'], index=7, help='The frequency of data points.')

# Slider for the percentage of training data
train_split = st.slider('Training Data Percentage', min_value=10, max_value=90, value=st.session_state.train_split, step=1, help='The percentage of data used for training.')

# Calendar inputs for start and end dates
start_date_calendar = st.date_input('Start Date (Calendar)', value=st.session_state.start_date, help='The start date for fetching historical data.')
end_date_calendar = st.date_input('End Date (Calendar)', value=st.session_state.end_date, help='The end date for fetching historical data.')

# Slider input for number of days
days_range = st.slider('Number of Days', min_value=1, max_value=60, value=st.session_state.days_range, step=1, help='The number of days for fetching historical data.')

# Calculate start_date and end_date based on days_range
end_date_slider = date.today()
start_date_slider = end_date_slider - timedelta(days=days_range)

# Determine which input to use (calendar or slider)
use_calendar = st.checkbox('Use Calendar Inputs', value=True)

# Set start_date and end_date based on the selected input method
if use_calendar:
    start_date = start_date_calendar
    end_date = end_date_calendar
else:
    start_date = start_date_slider
    end_date = end_date_slider

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
            data = yf.download(ticker, start=start_date, end=end_date + timedelta(days=1), interval=interval)  # Adjust end date
            if data.empty:
                st.error(f"No data fetched for {ticker}. Please check the ticker symbol or date range.")
                continue
            data = calculate_strategy_returns(data, entry_rsi, exit_rsi, window)
            plot_stock_and_rsi_strategy(data, ticker, entry_rsi, exit_rsi, window)
        except Exception as e:
            st.error(f"Error processing {ticker}:")
            st.error(traceback.format_exc())

if optimize_button:
    tickers_list = [ticker.strip().upper() for ticker in tickers.split(',')]
    for ticker in tickers_list:
        if not ticker:
            continue
        try:
            best_entry_rsi, best_exit_rsi, best_window, best_return = optimize_rsi(ticker, start_date, end_date, interval, train_split)
            if best_entry_rsi is not None and best_exit_rsi is not None and best_window is not None:
                st.success(f"{ticker} - Optimal Entry RSI: {best_entry_rsi}, Optimal Exit RSI: {best_exit_rsi}, Optimal Window: {best_window}, Best Return: {best_return * 100:.2f}%")
                st.session_state.entry_rsi = best_entry_rsi
                st.session_state.exit_rsi = best_exit_rsi
                st.session_state.window = best_window
                data = yf.download(ticker, start=start_date, end=end_date + timedelta(days=1), interval=interval)  # Adjust end date
                if data.empty:
                    st.error(f"No data fetched for {ticker}. Please check the ticker symbol or date range.")
                    continue
                data = calculate_strategy_returns(data, best_entry_rsi, best_exit_rsi, best_window)
                plot_stock_and_rsi_strategy(data, ticker, best_entry_rsi, best_exit_rsi, best_window)
        except Exception as e:
            st.error(f"Error optimizing {ticker}:")
            st.error(traceback.format_exc())
