import yfinance as yf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

class portfolio_optimisation:
    def __init__(self, tickers, num_portfolios) -> None:
        self.tickers = tickers
        self.num_portfolios = num_portfolios
        
        self.get_data()
        self.calculate_portfolios()
        self.present_weights()
    
    def get_data(self):
        self.data = yf.download(self.tickers, start='2010-01-01')["Adj Close"]
    
    def annual_returns(self, weights, mean_returns, cov_matrix):
        returns = np.sum(mean_returns*weights) *252
        std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        return std, returns

    def random_portfolios(self, num_portfolios, mean_returns, cov_matrix, risk_free_rate):
        # generate an empty list
        results = np.zeros((12,num_portfolios))
        # record the weights used
        weights_record = []
        
        for i in range(num_portfolios):
            # generate random values for the length of the data frame
            weights = np.random.random(len(self.data.columns))
            # standardise by the sum of weights
            weights /= np.sum(weights)
            # record the weights
            weights_record.append(weights)
            
            # calculate the volatility and returns
            portfolio_std_dev, portfolio_return = self.annual_returns(weights, mean_returns, cov_matrix)
            results[0,i] = portfolio_std_dev
            results[1,i] = portfolio_return
            # calculate the sharpe Ratio
            results[2,i] = (portfolio_return - risk_free_rate) / portfolio_std_dev   
        return results, weights_record
    
    def calculate_portfolios(self):
        print("Calculating portfolios...")
        
        returns = self.data.pct_change()
        mean_returns = returns.mean()   
        cov_matrix = returns.cov()
        
        risk_free_rate = yf.download('^TNX')['Close'][-1] / 100
        
        results, weights = self.random_portfolios(self.num_portfolios, mean_returns, cov_matrix, risk_free_rate)
        
        # calculate maximum sharpe ratio
        max_sharpe_idx = np.argmax(results[2])
        self.sdp, self.rp = results[0,max_sharpe_idx], results[1,max_sharpe_idx]
        # create data frame
        max_sharpe_allocation = pd.DataFrame(weights[max_sharpe_idx],index=self.data.columns,columns=['allocation'])
        # round the numbers
        max_sharpe_allocation.allocation = [round(i*100,2)for i in max_sharpe_allocation.allocation]
        self.max_sharpe_allocation = max_sharpe_allocation.T
        
        # calculate minimum volatility
        min_vol_idx = np.argmin(results[0])
        self.sdp_min, self.rp_min = results[0,min_vol_idx], results[1,min_vol_idx]
        # create data frame
        min_vol_allocation = pd.DataFrame(weights[min_vol_idx],index=self.data.columns,columns=['allocation'])
        # round the numbers
        min_vol_allocation.allocation = [round(i*100,2)for i in min_vol_allocation.allocation]
        self.min_vol_allocation = min_vol_allocation.T
        
        self.results = results
        self.weights = weights
        
    def present_weights(self):
        os.system("cls")
        print("Portfolio calculation done")
        print("-"*80)
        print("Maximum Sharpe Ratio Portfolio Allocation\n")
        print("Annualised Return:", round(self.rp,2))
        print("Annualised Volatility:", round(self.sdp,2))
        print("\n")
        print(self.max_sharpe_allocation)
        print("-"*80)
        print("Minimum Volatility Portfolio Allocation\n")
        print("Annualised Return:", round(self.rp_min,2))
        print("Annualised Volatility:", round(self.sdp_min,2))
        print("\n")
        print(self.min_vol_allocation)
        print("\n")
        
    def plot_results(self):
        plt.figure(figsize=(10, 7))
        plt.scatter(self.results[0,:], self.results[1,:], c=self.results[2,:],cmap='YlGnBu', marker='o', s=10, alpha=0.3)
        plt.colorbar()
        plt.scatter(self.sdp, self.rp,marker='*',color='r',s=300, label='Maximum Sharpe ratio')
        plt.scatter(self.sdp_min, self.rp_min,marker='*',color='g',s=300, label='Minimum volatility')
        plt.title('Simulated Portfolio Optimization based on Efficient Frontier')
        plt.xlabel('Annualised Volatility')
        plt.ylabel('Annualised Returns')
        plt.legend(labelspacing=0.8)
        plt.show()
        
tickers = ['SGEN', 'URI', 'WYNN', 'GEHC', 'LVS', 'STLD', 'BWA', 'ACGL', 'LW', 'RE']      

instance = portfolio_optimisation(tickers, 100000)
instance.plot_results()