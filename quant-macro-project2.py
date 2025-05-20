import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 対象国とファイル名
countries = {
    "Japan": "Japan_PWT1990.csv",
    "United States": "United_States_PWT1990.csv",
    "Germany": "Germany_PWT1990.csv",
    "Korea, Rep.": "Korea_Rep._PWT1990.csv"
}

alpha = 0.33  # 資本の所得分配率
results = []

for name, file in countries.items():
    df = pd.read_csv(file)
    df = df.sort_values('year')
    
    df['GDP_growth'] = np.log(df['cgdpo']).diff()
    df['K_growth'] = np.log(df['ck']).diff()
    df['L_growth'] = np.log(df['emp']).diff()
    df['TFP_growth'] = df['GDP_growth'] - alpha * df['K_growth'] - (1 - alpha) * df['L_growth']
    df['country'] = name
    results.append(df[['year', 'country', 'GDP_growth', 'K_growth', 'L_growth', 'TFP_growth']])

# 結合
all_data = pd.concat(results)

# プロット
plt.figure(figsize=(12, 6))
for country in countries.keys():
    data = all_data[all_data['country'] == country]
    plt.plot(data['year'], data['TFP_growth'], label=country)

plt.title("TFP Growth Rate by Country (1990–)")
plt.xlabel("Year")
plt.ylabel("Growth Rate (log-diff)")
plt.legend()
plt.grid(True)
plt.show()

