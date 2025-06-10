#hcを含んだバージョン
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
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

# 使用カラムにhc(人的資本指数)を追加
cols = ['country', 'year', 'rgdpna', 'rkna', 'emp', 'avh', 'labsh', 'hc']
data = pwt90[pwt90['country'].isin(oecd_countries) & pwt90['year'].between(1990, 2019)]
data = data[cols].dropna()

# 人的資本で補正された労働者数（効率的労働者）
data['eff_labor'] = data['emp'] * data['hc']

# 効率的労働者一人あたりのGDPと資本
data['y_per_effworker'] = data['rgdpna'] / data['eff_labor']
data['k_per_effworker'] = data['rkna'] / data['eff_labor']

# 資本シェア
data['alpha'] = 1 - data['labsh']

# 成長会計の関数（効率的労働者あたり）
def calc_eff_growth(df):
    df = df.sort_values('year')
    year_diff = df['year'].iloc[-1] - df['year'].iloc[0]

    g_y = np.log(df['y_per_effworker'].iloc[-1] / df['y_per_effworker'].iloc[0]) / year_diff * 100
    g_k = np.log(df['k_per_effworker'].iloc[-1] / df['k_per_effworker'].iloc[0]) / year_diff * 100
    alpha_mean = df['alpha'].mean()

    # Solow残差
    g_a = g_y - alpha_mean * g_k
    cap_deep = alpha_mean * g_k
    tfp_share = g_a / g_y if g_y != 0 else np.nan
    cap_share = cap_deep / g_y if g_y != 0 else np.nan

    return pd.Series({
        'Growth Rate': round(g_y, 2),
        'TFP Growth': round(g_a, 2),
        'Capital Deepening': round(cap_deep, 2),
        'TFP Share': round(tfp_share, 2),
        'Capital Share': round(cap_share, 2)
    })

# 国別に計算
results = data.groupby('country').apply(calc_eff_growth).reset_index()
avg_row = results.mean(numeric_only=True).round(2).to_dict()
avg_row['country'] = 'Average'
results = pd.concat([results, pd.DataFrame([avg_row])], ignore_index=True)


# 表の表示設定 (Table5.1を再現、長すぎるため関数にしてトグル表示)
def display_growth_table(results):
    fig, ax = plt.subplots(figsize=(10, len(results)*0.3 + 2.0))
    ax.axis('off')

    # タイトル
    ax.text(0, 0.8, 'Growth Accounting in OECD Countries: 1990-2019(HC considered)', transform=ax.transAxes,
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



