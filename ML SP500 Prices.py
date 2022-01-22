import bs4 as bs  # web scraping, turns source code into treatable object
import datetime as dt  # specifies dates for datareader
import os, pickle, requests
import pandas as pd
import pandas_datareader.data as web
import yfinance as yf #figure out IMPORT

import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

from collections import Counter  # visualize distributions of classes both in dataset and algo predictions

# machine learning frameworks
from sklearn import svm, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split

style.use('ggplot')

yf.pdr_override()


def save_sp500_tickers():
    ''''Downloads list of SP500 companies'''

    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')  # access source code
    soup = bs.BeautifulSoup(resp.text, 'lxml')  # turns text into object
    table = soup.find('table', {'class': 'wikitable sortable'})  # finds table of stock data
    tickers = []

    # makes list of tickers
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text.replace('.', '-')
        ticker = ticker[:-1]
        tickers.append(ticker)

    # dumps tickers into a reusable file
    with open("sp500tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)

    # print(tickers)
    return tickers


def get_data_from_yahoo(reload_sp500=False):
    '''Downloads stock pricing data'''

    if reload_sp500:  # if needed, re-pull SP500 list
        tickers = save_sp500_tickers()
    else:  # use pickle file
        with open("sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)

    if not os.path.exists('stock_dfs'):  # make dir to save stock pricing data
        os.makedirs('stock_dfs')

    start = dt.datetime(2019, 6, 8)
    end = dt.datetime.now()

    for ticker in tickers:  # pulls the data and stores it in the dir
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):  # if we don't have the stock file
            df = web.get_data_yahoo(ticker, start, end)  # dataframe of sotck ohlcv
            df.reset_index(inplace=True)  # inplace=True used to not redefine df each time
            df.set_index("Date", inplace=True)
            df = df.drop("Symbol", axis=1)
            df.to_csv('stock_dfs/{}.csv'.format(ticker))
        else:  # if we have the stock file
            print('Already have{}'.format(ticker))


def compile_data():
    '''Combines SP500 list and stock prices in one DataFrame'''

    with open("sp500tickers.pickle", "rb") as f:  # opening list
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):  # reading each stock's dataframe
        df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
        df.set_index('Date', inplace=True)

        df.rename(columns={'Adj Close': ticker}, inplace=True)
        df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], 1, inplace=True)

        if main_df.empty:  # if main dataframe empty then start with current one
            main_df = df
        else:  # otherwise join them
            main_df = main_df.join(df, how='outer')

        if count % 10 == 0:  # output count of the current ticker if it's evenly divisible by 10
            print(count)

    print(main_df.head())
    main_df.to_csv('sp500_joined_closes.csv')


def visualize_data():
    '''Plots a correlation table between each stock'''

    df = pd.read_csv('sp500_joined_closes.csv')
    df_corr = df.corr()  # ☺determine correlation of every column to every column
    print(df_corr.head())

    # create heatmap to visualize correlations
    data1 = df_corr.values  # get values
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)

    # create heatmap and add colorbar as a scale
    heatmap1 = ax1.pcolor(data1, cmap=plt.cm.RdYlGn)
    fig1.colorbar(heatmap1)

    # create and rearrange axis
    ax1.set_xticks(np.arange(data1.shape[1]) + 0.5, minor=False)
    ax1.set_yticks(np.arange(data1.shape[0]) + 0.5, minor=False)
    ax1.invert_yaxis()  # now stocks are from A to Z
    ax1.xaxis.tick_top()  # put x axis to the top

    # add company names
    column_labels = df_corr.columns
    row_labels = df_corr.index
    ax1.set_xticklabels(column_labels)
    ax1.set_yticklabels(row_labels)

    plt.xticks(rotation=90)
    heatmap1.set_clim(-1, 1)  # colormap ranges from -1 to 1
    plt.tight_layout()
    # plt.savefig("correlations.png", dpi=(300))
    plt.show()


def process_data_for_labels(ticker):
    '''Processes data for labels, in this case changes in value over 7 days
       for a SINGLE stock'''

    hm_days = 7  # future prices for the 7 days
    df = pd.read_csv('sp500_joined_closes.csv', index_col=0)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)

    # take % change values for the next 7 days
    for i in range(1, hm_days + 1):
        df['{}_{}d'.format(ticker, i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]

    df.fillna(0, inplace=True)
    return tickers, df


def buy_sell_hold(*args):  # args = future price changes columns
    '''Makes the decision to either buy, sell or hold a stock'''

    cols = [c for c in args]
    requirement = 0.02

    for col in cols:
        if col > requirement:
            return 1  # buy
        if col < -requirement:
            return -1  # sell

    return 0  # hold


def extract_featuresets(ticker):
    '''Creates needed dataset and creates label'''
    '''Label : target column that has either -1, 0 or 1'''

    tickers, df = process_data_for_labels(ticker)

    df['{}_target'.format(ticker)] = list(map(buy_sell_hold,
                                              df['{}_1d'.format(ticker)],
                                              df['{}_2d'.format(ticker)],
                                              df['{}_3d'.format(ticker)],
                                              df['{}_4d'.format(ticker)],
                                              df['{}_5d'.format(ticker)],
                                              df['{}_6d'.format(ticker)],
                                              df['{}_7d'.format(ticker)]))

    # get the distribution
    vals = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    print('Data spread', Counter(str_vals))

    # clean up data
    df.fillna(0, inplace=True)  # replace missing data with 0
    df = df.replace([np.inf, -np.inf], np.nan)  # replace infinite data to NaNs
    df.dropna(inplace=True)  # drop infinite values

    '''Instead of using day's prices of stocks as features, we can use % change that day,
       that way we can profit of companies that lag behing those who already got their price changed'''

    # convert stock prices to % changes and clean up data
    df_vals = df[[ticker for ticker in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)

    # create Features and Labels
    X = df_vals.values
    y = df['{}_target'.format(ticker)].values
    return X, y, df


def do_ml(ticker):
    '''Machine Learning Function'''

    X, y, df = extract_featuresets(ticker)  # gets featuresets and labels
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.25)  # shuffle data and creates training and testing samples

    clf = VotingClassifier([('lsvc', svm.LinearSVC()),
                            ('knn', neighbors.KNeighborsClassifier()),
                            ('rfor', RandomForestClassifier())])  # choose a classifier
    clf.fit(X_train, y_train)  # fit X data to y data
    confidence = lf.score(X_test,
                           y_test)  # test results by making a prediction with X_test(featuresets) and see if it matches y_test(labels)

    print('accuracy:', confidence)
    predictions = clf.predict(X_test)
    print('predicted class counts:', Counter(predictions))

    return confidence





