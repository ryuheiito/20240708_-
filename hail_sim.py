import pandas as pd
import numpy as np
from tqdm import tqdm
import logging

# ログの設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# データの読み込み
try:
    hail_data = pd.read_csv('input/hail_range.csv')
    solar_data = pd.read_csv('input/solar_panels.csv')
except FileNotFoundError as e:
    logging.error(f"データの読み込みエラー: {e}")
    raise

def monte_carlo_simulation(prefecture, panel_name, iterations=10000, seed=42):
    """
    特定の都道府県とソーラーパネルに対するPMLを推定するためのモンテカルロシミュレーションを実行します。
    """
    np.random.seed(seed)
    
    hail_info = hail_data[hail_data['Prefecture'] == prefecture]
    solar_info = solar_data[(solar_data['Prefecture'] == prefecture) & (solar_data['PanelName'] == panel_name)]
    
    if hail_info.empty or solar_info.empty:
        logging.warning(f"データが見つかりません - 都道府県: {prefecture}, パネル: {panel_name}")
        return None, None, None, None, None, None, None
    
    hail_info = hail_info.iloc[0]
    solar_info = solar_info.iloc[0]
    
    min_hail_range = hail_info['MinHailRange']
    max_hail_range = hail_info['MaxHailRange']
    avg_hail_count = hail_info['AverageHailCount']
    mean_hail_size = hail_info['MeanHailSize']
    hail_size_std_dev = hail_info['HailSizeStdDev']
    solar_area = solar_info['Area']
    solar_cost = solar_info['Cost']

    damages = []
    
    for _ in range(iterations):
        max_damage = 0
        hail_count = np.random.poisson(avg_hail_count)
        for _ in range(hail_count):
            hail_size = np.random.normal(mean_hail_size, hail_size_std_dev)
            if hail_size >= 2.5:
                hail_range = np.random.uniform(min_hail_range, max_hail_range)
                damage = solar_cost if hail_range >= solar_area else solar_cost * (hail_range / solar_area)
                if damage > max_damage:
                    max_damage = damage
        damages.append(max_damage)
    
    damages.sort(reverse=True)
    PML_100 = damages[int(iterations / 100) - 1]
    PML_50 = damages[int(iterations / 50) - 1]
    PML_10 = damages[int(iterations / 10) - 1]
    
    PML_100_percent = (PML_100 / solar_cost) * 100 if solar_cost != 0 else 0
    PML_50_percent = (PML_50 / solar_cost) * 100 if solar_cost != 0 else 0
    PML_10_percent = (PML_10 / solar_cost) * 100 if solar_cost != 0 else 0
    
    return panel_name, PML_100, PML_50, PML_10, PML_100_percent, PML_50_percent, PML_10_percent

results = []

for prefecture in tqdm(solar_data['Prefecture'].unique(), desc="都道府県を処理中"):
    prefecture_solar_data = solar_data[solar_data['Prefecture'] == prefecture]
    for _, row in prefecture_solar_data.iterrows():
        panel_name = row['PanelName']
        result = monte_carlo_simulation(prefecture, panel_name)
        if result[0] is not None:
            results.append((prefecture, *result))

results_df = pd.DataFrame(results, columns=['都道府県', 'パネル名', 'PML_100', 'PML_50', 'PML_10', 'PML_100_割合', 'PML_50_割合', 'PML_10_割合'])
results_df.to_csv('pml_results.csv', index=False, encoding='utf-8-sig')

logging.info("シミュレーション結果が'pml_results.csv'に保存されました。")
