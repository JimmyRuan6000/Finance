#import the important libries that downloaded on command prompt

import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize

# Step 1: obtain the info of the whole S&P 500 tickers/company list from Wikipedia
url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
sp500_table = pd.read_html(url)[0]
tickers = sp500_table['Symbol'].tolist()

# Step 2: Get user to inputs the ticker and the total amount they are investing
budget = float(input("Enter your total budget in USD: "))
ticker_input = input("Enter the ticker of the company (must be in S&P 500): ").upper()
 # stop the code if the ticker input is not on the wikipedia/ S & p 500
if ticker_input not in tickers:
    print("Ticker not found in S&P 500 list.")
    exit()

# Step 3: Download historical stock data (last 5 years=252 days) from yahoo finance libry

data = yf.download(ticker_input, period="5y", auto_adjust=False,progress=False)


data['Return'] = data['Close'].pct_change()
avg_return = data['Return'].mean() * 252  # annualized return
volatility = data['Return'].std() * np.sqrt(252)  # annualized volatility

# Step 4:  risk and return assumptions for long vs short term
long_term_return = avg_return * 0.9
long_term_risk = volatility * 0.7
short_term_return = avg_return * 1.2
short_term_risk = volatility * 1.5

# Step 5: Optimization function [ maximize return use Sharpe ratio: (expected return-risk free of retun)/standard deviation of anual return and stock volatility ]
def portfolio_sharpe(weights):
    w_long, w_short = weights
    total_return = w_long * long_term_return + w_short * short_term_return
    total_risk = np.sqrt((w_long * long_term_risk) ** 2 + (w_short * short_term_risk) ** 2)
    return - (total_return / total_risk)

constraints = ({'type': 'eq', 'fun': lambda w: w[0] + w[1] - 1})
bounds = ((0, 1), (0, 1))

result = minimize(portfolio_sharpe, [0.5, 0.5], bounds=bounds, constraints=constraints)
w_long, w_short = result.x

# Step 6: print the sugguested percentage and number result to invest on the stock
amount_long = budget * w_long
amount_short = budget * w_short
print(f"\nOptimal Allocation for {ticker_input}:")
print(f"Long Term: ${amount_long:,.2f} ({w_long*100:.2f}%)")
print(f"Short Term: ${amount_short:,.2f} ({w_short*100:.2f}%)")
print(f"Expected annual return (based on history): {avg_return*100:.2f}%")
print(f"Expected annual volatility: {volatility*100:.2f}%")
