import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# PWT9.0 データの読み込み
pwt90 = pd.read_stata('https://www.rug.nl/ggdc/docs/pwt90.dta')

# Table5.1よりOECD諸国
oecd_countries = [
    'Australia', 'Austria', 'Belgium', 'Canada', 'Denmark', 'Finland', 'France',
    'Germany', 'Greece', 'Iceland', 'Ireland', 'Italy', 'Japan', 'Netherlands',
    'New Zealand', 'Norway', 'Portugal', 'Spain', 'Sweden', 'Switzerland',
    'United Kingdom', 'United States'
]

# 分析対象期間（1990-2019）
data = pwt90[
    pwt90['country'].isin(oecd_countries) &
    pwt90['year'].between(1990, 2019)
]

# 用いるカラム
# country: 国名
# year: 年
# rgdpna: 実質GDP (Y)
# rkna: 実質資本ストック (K)
# pop: 人口
# emp: 就業者数
# avh: 就業者一人当たりの年間平均労働時間
# labsh: 労働分配率 [0,1]
# rtfpna: 効率的生産性(全要素生産性TFPインデックス)
cols = ['country', 'year', 'rgdpna', 'rkna', 'pop', 'emp', 'avh', 'labsh', 'rtfpna']
data = data[cols].dropna()

# 労働時間とα（資本分配率）の計算
data['hours'] = data['emp'] * data['avh'] #労働を表すものとして今回は就業者数と労働時間の積[h]を使用
data['alpha'] = 1 - data['labsh']

# 成長会計の計算関数
def calculate_growth_rates(df):
    df = df.sort_values('year')
    year_diff = df['year'].iloc[-1] - df['year'].iloc[0]
    
    g_y = (np.log(df['rgdpna'].iloc[-1]/df['rgdpna'].iloc[0])) / year_diff * 100  # 実質GDPの成長率
    g_k = (np.log(df['rkna'].iloc[-1]/df['rkna'].iloc[0])) / year_diff * 100      # 実質資本ストックの成長率
    g_a = (np.log(df['rtfpna'].iloc[-1]/df['rtfpna'].iloc[0])) / year_diff * 100  # TFPの成長率
    g_l = (np.log(df['hours'].iloc[-1]/df['hours'].iloc[0])) / year_diff * 100    # 労働（就業者数×労働時間）の成長率

    alpha_mean = df['alpha'].mean()
    cap_deep = alpha_mean * (g_k - g_l)  

    tfp_share = g_a / g_y if g_y != 0 else np.nan
    cap_share = g_k / g_y if g_y != 0 else np.nan

    return pd.Series({
        'Growth Rate': round(g_y, 2),
        'TFP Growth': round(g_a, 2),
        'Capital Deepening': round(cap_deep, 2),
        'TFP Share': round(tfp_share, 2),
        'Capital Share': round(cap_share, 2)
    })

#国別に成長会計を計算
results = data.groupby('country').apply(calculate_growth_rates).reset_index() 

# 平均行の追加
avg_row = results.mean(numeric_only=True).round(2).to_dict()
avg_row['country'] = 'Average'
results = pd.concat([results, pd.DataFrame([avg_row])], ignore_index=True)

# 表の表示設定 (Table5.1を再現、長すぎるため関数にしてトグル表示)
def display_growth_table(results):
    fig, ax = plt.subplots(figsize=(10, len(results)*0.3 + 2.0))
    ax.axis('off')

    # タイトル
    ax.text(0, 0.8, 'Growth Accounting in OECD Countries: 1990-2019', transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='bottom', ha='left')

    # 表の作成
    table = ax.table(
        cellText=results.round(2).values,
        colLabels=results.columns,
        loc='center',
        cellLoc='center',
        colLoc='center',
        edges='open'  
    )

    # フォントサイズと行間調整
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    # 正しい行インデックス指定
    header_row = 0
    average_row = len(results)

    # セルデザインの調整
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0) 
        cell.set_edgecolor('black')
        cell.PAD = 2.0  

        if row == header_row:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#f2f2f2')
            cell.set_linewidth(0.8)
            cell.visible_edges = 'T'
            cell.visible_edges += 'B'

        if row == average_row:
            cell.set_linewidth(0.8)
            cell.visible_edges = 'T'
            cell.visible_edges += 'B'

    plt.show()

# 表を出力
display_growth_table(results)



