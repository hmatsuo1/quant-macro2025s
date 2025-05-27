
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
import matplotlib.font_manager as fm

# データ取得期間
start_date = '1955-01-01'
end_date = '2022-01-01'

# 1.今回はイタリアを選択、日本とイタリアの実質GDPデータをFREDから取得
japan_gdp = web.DataReader('JPNRGDPEXP', 'fred', start_date, end_date)
italy_gdp = web.DataReader('CLVMNACSCAB1GQIT', 'fred', start_date, end_date)

# 対数を取る
log_japan_gdp = np.log(japan_gdp)
log_italy_gdp = np.log(italy_gdp)

# 2.HPフィルターを適用（λ=1600,cycle:循環変動成分, trend:トレンド成分）
cycle_japan, trend_japan = sm.tsa.filters.hpfilter(log_japan_gdp, lamb=1600)
cycle_italy, trend_italy = sm.tsa.filters.hpfilter(log_italy_gdp, lamb=1600)

# 3.循環変動成分の標準偏差を計算
std_japan = cycle_japan.std()
std_italy = cycle_italy.std()

# 相関係数を計算
corr = cycle_japan.corr(cycle_italy)

# 日本語フォント設定
font_path = "C:/Windows/Fonts/msgothic.ttc"
jp_font = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = jp_font.get_name()

# 結果を表示
results = pd.DataFrame({
    '標準偏差': [std_japan, std_italy],
    '日本との相関係数': [1.0, corr]
}, index=['日本', 'イタリア']).round(4)
print(results)

# 表の出力
fig, ax = plt.subplots(figsize=(6, 2.5))
ax.axis('off')
table = ax.table(cellText=results.values,
                 colLabels=results.columns,
                 rowLabels=results.index,
                 cellLoc='center',
                 loc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.4, 1.4)
plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
plt.title('両国の標準偏差と相関係数', pad=20)
plt.show()

# 循環変動成分の時系列データをプロット
plt.figure(figsize=(12,6))
plt.plot(cycle_japan, label='日本')
plt.plot(cycle_italy, label='イタリア')
plt.legend()
plt.title('日本とイタリアの実質GDPの循環変動成分')
plt.xlabel('年')
plt.ylabel('対数GDP（循環変動成分）')
plt.tight_layout()
plt.show()
