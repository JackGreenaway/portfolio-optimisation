import nashpy as nash
import numpy as np
import yfinance as yf

class gametheory_optimisation:
    def __init__(self, tickers) -> None:
        self.tickers = tickers
        
        # methods
        self.get_data()
        
    def get_data(self):
        raw_data = yf.download(self.tickers, start="2010-01-01")["Adj Close"]
    
    def risk_returns(self):
        pass
    
if __name__ == "__main__":
    tickers = ["AAPL", "AMZN", "MSFT", "LMT", "META", "GOOG"]
    gametheory_optimisation(tickers=tickers)