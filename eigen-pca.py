"Constructed using research from the following:"

"Avellaneda, M., & Lee, J.-H. (2010). Statistical arbitrage in the us equities market. Quantitative Finance, 10(7),pp. 761–761."
"A. Xiong and A. N. Akansu (2019). Performance Comparison of Minimum Variance, Market and Eigen Portfolios for US Equities.  53rd Annual Conference on Information Sciences and Systems (CISS), 2019, pp. 1-5, doi: 10.1109/CISS.2019.8693035."

from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import os

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)


class pca_optimisation:
    def __init__(self, tickers) -> None:
        self.tickers = tickers

        # initialise methods
        self.get_data()
        self.format_data()
        self.fit_pca()
        self.plot_pca()
        self.pca_weights()

    def get_data(self):
        # simply download the data
        raw_data = yf.download(self.tickers, start="2020-01-01")["Adj Close"]

        # add the market return onto the end of the dataframe
        raw_data["market"] = yf.download("^GSPC", start="2020-01-01")["Adj Close"]

        raw_data = raw_data.dropna(axis=1)
        self.raw_data = raw_data

    def format_data(self):
        # we need to normalise the dataset

        # daily linear returns
        self.asset_returns = np.log(self.raw_data / self.raw_data.shift(1)).dropna()
        # normalised returns
        self.norm_returns = (
            self.asset_returns - self.asset_returns.mean()
        ) / self.asset_returns.std()

        # create train and test dataset
        # 90% = training, 10% = test
        self.df_train = self.norm_returns[: int(len(self.norm_returns) * 0.95)]
        self.df_test = self.norm_returns[int(len(self.norm_returns) * 0.95) :]

        self.df_train_raw = self.asset_returns[: int(len(self.asset_returns) * 0.95)]
        self.df_test_raw = self.asset_returns[int(len(self.asset_returns) * 0.95) :]

    def fit_pca(self):
        os.system("cls")
        # n_tickers = self.norm_returns.columns.values[:-1]

        # remove the market from the end of the dataframe
        self.stock_tickers = self.asset_returns.columns.values[:-1]

        # normalised returns
        cov_norm = self.df_train[self.stock_tickers].cov()
        self.pca_norm = PCA()
        self.pca_norm.fit(cov_norm)

        var_threshold = 0.90
        var_explained_norm = np.cumsum(self.pca_norm.explained_variance_ratio_)
        num_comp_norm = (
            np.where(np.logical_not(var_explained_norm < var_threshold))[0][0] + 1
        )  # +1 due to zero based-arrays
        print(
            "%d components explain %.2f%% of variance (NORMALISED)"
            % (num_comp_norm, 100 * var_threshold)
        )

        # linear returns
        cov_linear = self.df_train_raw[self.stock_tickers].cov()
        pca_linear = PCA()
        pca_linear.fit(cov_linear)

        var_explained_linear = np.cumsum(pca_linear.explained_variance_ratio_)
        num_comp_linear = (
            np.where(np.logical_not(var_explained_linear < var_threshold))[0][0] + 1
        )  # +1 due to zero based-arrays
        print(
            "%d components explain %.2f%% of variance (LINEAR)"
            % (num_comp_linear, 100 * var_threshold)
        )

    def plot_pca(self):
        # plot the pca for the normalised returns
        bar_width = 0.9

        if len(self.pca_norm.components_) > 100:
            n_asset = int(len(self.pca_norm.components_) * 0.05)
        elif len(self.pca_norm.components_) > 50:
            n_asset = int(len(self.pca_norm.components_) * 0.2)
        else:
            n_asset = int(len(self.pca_norm.components_) * 0.75)

        x_indx = np.arange(n_asset)
        fig, ax = plt.subplots()
        fig.set_size_inches(12, 4)
        # Eigenvalues are measured as percentage of explained variance.
        rects = ax.bar(
            x_indx,
            self.pca_norm.explained_variance_ratio_[:n_asset],
            bar_width,
            color="deepskyblue",
        )
        ax.set_xticks(x_indx + bar_width / 2)
        ax.set_xticklabels(list(range(n_asset)), rotation=45)
        ax.set_title("Percent variance explained (normalised returns)")
        ax.legend((rects[0],), ("Percent variance explained by principal components",))
        plt.show()

    def pca_weights(self):
        pcs = self.pca_norm.components_
        normalised_pcs = []

        for vector in pcs:
            normalised_value = vector / vector.sum()
            normalised_pcs.append(normalised_value)

        pca_weight = normalised_pcs[0]

        eigen_prtf1 = pd.DataFrame(
            data={"weights": pca_weight.squeeze()}, index=self.stock_tickers
        )

        eigen_prtf1.sort_values(by=["weights"], ascending=False, inplace=True)

        eigen_prtf1.plot(
            title="Normalised eigen-portfolio weights",
            figsize=(12, 6),
            xticks=range(0, len(self.stock_tickers)),
            rot=45,
            linewidth=3,
        )

        print(f"\n{'***' * 15} Portfolio Weights {'***' * 15}\n")
        print(f"{eigen_prtf1.T.head(len(self.stock_tickers))}")
        print("\nSum of weights of eigen-portfolio: %.2f\n" % np.sum(eigen_prtf1))

        plt.show()

    def backtest(self):
        pass


if __name__ == "__main__":
    tickers = [
        "AAPL",
        "META",
        "AMZN",
        "GOOG",
        "GS",
        "XOM",
        "RBLX",
        "GME",
        "FRGT",
        "GTLB",
        "WFC",
        "NVDA",
        "JPM",
        "TSLA",
        "PD",
        "JNJ",
        "LMH",
        "BBBY",
        "COST",
        "SI",
        "RKT",
    ]
    sp = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    sp = list(sp.Symbol)

    # initialise class
    instance = pca_optimisation(tickers=tickers)
