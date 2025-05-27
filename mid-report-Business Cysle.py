import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas_datareader as pdr
import numpy as np

# 期間設定
start_date = '2000-01-01'
end_date = '2022-01-01'

# 今回はイタリアを選択、日本とイタリアについて分析を行う
gdp_japan = web.DataReader('JPNRGDPEXP', 'fred', start_date, end_date)   # 日本
gdp_italy = web.DataReader('NGDPRSAXDCITQ', 'fred', start_date, end_date) # イタリア


gdp_data = {
    'Japan': np.log(gdp_japan),
    'Italy': np.log(gdp_italy),
}

# Lambda values to apply
lambdas = [10, 100, 1600]
colors = {10: 'red', 100: 'orange', 1600: 'purple'}
# Plotting

for country, log_gdp in gdp_data.items():
    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot log GDP (actual data)
    ax.plot(log_gdp.index, log_gdp, label='Log GDP (Actual)', color='blue', linewidth=1.5)

    # Plot trends for each lambda
    for lamb in lambdas:
        _, trend = sm.tsa.filters.hpfilter(log_gdp.squeeze(), lamb=lamb)
        ax.plot(trend.index, trend, label=f'Trend (lambda={lamb})', color=colors[lamb], linestyle='--')

    ax.set_xlabel('Date')
    ax.set_ylabel('Log GDP and Trend')
    ax.set_title(f'{country} - Log GDP and HP Filter Trends (λ=10, 100, 1600)')
    ax.legend(loc='best')
    ax.grid(True)

    plt.show()