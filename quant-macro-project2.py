import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# αの設定（資本分配率）
alpha = 0.33

# データ読み込み（例: PWTからDLしたCSV）
df = pd.read_csv("pwt_data.csv")

# 必要列を抽出してフィルタリング（例: 1990年以降）
df = df[df['year'] >= 1990]
df = df[df['country'].isin(['Japan', 'United States', 'Germany', 'Korea, Rep.'])]

# 各国ごとに成長率とTFP成長率を計算
def compute_growth(group):
    group = group.sort_values('year')
    group['GDP_growth'] = np.log(group['cgdpo']).diff()
    group['K_growth'] = np.log(group['ck']).diff()
    group['L_growth'] = np.log(group['emp']).diff()
    group['TFP_growth'] = group['GDP_growth'] - alpha * group['K_growth'] - (1 - alpha) * group['L_growth']
    return group

df = df.groupby('country').apply(compute_growth)

# 結果を確認
print(df[['country', 'year', 'GDP_growth', 'K_growth', 'L_growth', 'TFP_growth']].dropna().head())

# TFP成長率のグラフ
plt.figure(figsize=(12, 6))
for country in df['country'].unique():
    plt.plot(df[df['country'] == country]['year'], df[df['country'] == country]['TFP_growth'], label=country)

plt.title("TFP Growth Rate (1990–Present)")
plt.xlabel("Year")
plt.ylabel("Growth Rate (log-difference)")
plt.legend()
plt.grid(True)
plt.show()
