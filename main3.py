import pandas as pd
import json
from datetime import datetime
import glob
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_product_catalog(catalog_path):
    try:
        with open(catalog_path, 'r', encoding='utf-8') as f:
            catalog_data = json.load(f)
        products = catalog_data.get('products', [])
        id_to_info = {}
        for product in products:
            item_id = str(product.get('id'))
            if item_id:
                id_to_info[item_id] = {
                    'category': product.get('category', '未分类'),
                    'price': float(product.get('price', 0))
                }
        if not id_to_info:
            raise ValueError("没有找到有效的商品数据")
        return id_to_info
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON解析错误: {str(e)}")
    except Exception as e:
        raise ValueError(f"读取商品目录文件时出错: {str(e)}")

def get_main_category(sub_category):
    category_mapping = {
        '智能手机': '电子产品', '笔记本电脑': '电子产品', '平板电脑': '电子产品', '智能手表': '电子产品',
        '耳机': '电子产品', '音响': '电子产品', '相机': '电子产品', '摄像机': '电子产品', '游戏机': '电子产品',
        '上衣': '服装', '裤子': '服装', '裙子': '服装', '内衣': '服装', '鞋子': '服装', '帽子': '服装',
        '手套': '服装', '围巾': '服装', '外套': '服装',
        '零食': '食品', '饮料': '食品', '调味品': '食品', '米面': '食品', '水产': '食品', '肉类': '食品',
        '蛋奶': '食品', '水果': '食品', '蔬菜': '食品',
        '家具': '家居', '床上用品': '家居', '厨具': '家居', '卫浴用品': '家居',
        '文具': '办公', '办公用品': '办公',
        '健身器材': '运动户外', '户外装备': '运动户外',
        '玩具': '玩具', '模型': '玩具', '益智玩具': '玩具',
        '婴儿用品': '母婴', '儿童课外读物': '母婴',
        '车载电子': '汽车用品', '汽车装饰': '汽车用品'
    }
    return category_mapping.get(sub_category, '其他')

def extract_time_and_categories(purchase_history, id_to_info):
    try:
        if isinstance(purchase_history, str):
            purchase_data = json.loads(purchase_history)
        else:
            purchase_data = purchase_history
        purchase_date = datetime.strptime(purchase_data.get('purchase_date', ''), '%Y-%m-%d')
        categories = set()
        for item in purchase_data.get('items', []):
            item_id = str(item.get('id'))
            if item_id in id_to_info:
                sub_category = id_to_info[item_id]['category']
                main_category = get_main_category(sub_category)
                categories.add(main_category)
        return purchase_date, categories
    except Exception as e:
        print(f"解析购买历史时出错: {str(e)}")
        return None, set()

def analyze_time_patterns(
    parquet_dir,
    product_catalog_path,
    batch_size=1000000,
    columns=None,
    sample_ratio=1.0
):
    def load_parquet_files(parquet_dir, columns=None, batch_size=1000000, sample_ratio=1.0):
        if not 0 < sample_ratio <= 1:
            raise ValueError("抽样比例必须在(0, 1]范围内")
        parquet_files = glob.glob(os.path.join(parquet_dir, "*.parquet"))
        parquet_files = parquet_files[:1]
        if not parquet_files:
            raise ValueError(f"在目录 {parquet_dir} 中没有找到parquet文件")
        all_data = []
        for file_path in parquet_files:
            try:
                df = pd.read_parquet(file_path, columns=columns)
                if sample_ratio < 1:
                    df = df.sample(frac=sample_ratio, random_state=42)
                total_rows = len(df)
                if total_rows > batch_size:
                    for start_idx in range(0, total_rows, batch_size):
                        end_idx = min(start_idx + batch_size, total_rows)
                        batch_df = df.iloc[start_idx:end_idx]
                        all_data.append(batch_df)
                else:
                    all_data.append(df)
            except Exception as e:
                print(f"读取文件 {file_path} 时出错: {str(e)}")
                continue
        if not all_data:
            raise ValueError("没有成功读取任何数据")
        final_df = pd.concat(all_data, ignore_index=True)
        return final_df
    try:
        id_to_info = load_product_catalog(product_catalog_path)
        df = load_parquet_files(parquet_dir, columns=columns, batch_size=batch_size, sample_ratio=sample_ratio)
        seasonal_patterns = defaultdict(lambda: defaultdict(int))
        monthly_patterns = defaultdict(lambda: defaultdict(int))
        weekly_patterns = defaultdict(lambda: defaultdict(int))
        category_time_series = defaultdict(list)
        sequential_patterns = defaultdict(lambda: defaultdict(int))
        purchase_records = []
        for purchase_history in tqdm(df['purchase_history'], desc="处理时间序列数据", ncols=100):
            purchase_date, categories = extract_time_and_categories(purchase_history, id_to_info)
            if purchase_date and categories:
                purchase_records.append((purchase_date, categories))
                quarter = f"Q{(purchase_date.month-1)//3 + 1}"
                month = purchase_date.strftime('%m')
                weekday = purchase_date.strftime('%A')
                for category in categories:
                    seasonal_patterns[quarter][category] += 1
                    monthly_patterns[month][category] += 1
                    weekly_patterns[weekday][category] += 1
                    category_time_series[category].append(purchase_date)
        purchase_records.sort(key=lambda x: x[0])
        for i in range(len(purchase_records)-1):
            current_date, current_categories = purchase_records[i]
            next_date, next_categories = purchase_records[i+1]
            if (next_date - current_date).days <= 30:
                for cat1 in current_categories:
                    for cat2 in next_categories:
                        if cat1 != cat2:
                            sequential_patterns[cat1][cat2] += 1
        category_time_features = {}
        for category, timestamps in category_time_series.items():
            if timestamps:
                df_category = pd.Series(timestamps)
                category_time_features[category] = {
                    'total_purchases': len(timestamps),
                    'peak_month': df_category.dt.month.mode().iloc[0],
                    'peak_weekday': df_category.dt.dayofweek.mode().iloc[0],
                    'average_interval': df_category.diff().mean().days if len(timestamps) > 1 else None
                }
        return {
            'seasonal_patterns': dict(seasonal_patterns),
            'monthly_patterns': dict(monthly_patterns),
            'weekly_patterns': dict(weekly_patterns),
            'sequential_patterns': dict(sequential_patterns),
            'category_time_features': category_time_features,
            'total_records': len(purchase_records)
        }
    except Exception as e:
        raise Exception(f"时间序列分析过程中出错: {str(e)}")

def print_time_analysis(result):
    print("\n=== 时间序列模式分析结果 ===")
    total_records = result['total_records']
    print("\n1. 季节性模式:")
    seasonal_df = pd.DataFrame(result['seasonal_patterns']).fillna(0)
    seasonal_df = (seasonal_df / seasonal_df.sum().sum() * 100).round(2)
    print("\n每个季度的热门类别（前5）及其占比(%):")
    for quarter in seasonal_df.columns:
        top_categories = seasonal_df[quarter].sort_values(ascending=False).head(5)
        print(f"\n{quarter}:")
        print(top_categories.apply(lambda x: f"{x:.2f}%"))
    print("\n2. 月度模式:")
    monthly_df = pd.DataFrame(result['monthly_patterns']).fillna(0)
    monthly_total = monthly_df.sum().sum()
    monthly_proportions = (monthly_df.sum() / monthly_total * 100).round(2)
    print("\n每月的购买频率占比(%):")
    print(monthly_proportions.apply(lambda x: f"{x:.2f}%"))
    print("\n3. 周度模式:")
    weekly_df = pd.DataFrame(result['weekly_patterns']).fillna(0)
    weekly_total = weekly_df.sum().sum()
    weekly_proportions = (weekly_df.sum() / weekly_total * 100).round(2)
    print("\n每周各日的购买频率占比(%):")
    print(weekly_proportions.apply(lambda x: f"{x:.2f}%"))
    print("\n4. 类别时间特征:")
    total_purchases = sum(features['total_purchases'] for features in result['category_time_features'].values())
    for category, features in result['category_time_features'].items():
        print(f"\n{category}:")
        purchase_percentage = (features['total_purchases'] / total_purchases * 100)
        print(f"  购买频率占比: {purchase_percentage:.2f}%")
        print(f"  峰值月份: {features['peak_month']}月")
        print(f"  峰值星期: 星期{features['peak_weekday']+1}")
        if features['average_interval']:
            print(f"  平均购买间隔: {features['average_interval']:.1f}天")
    print("\n5. 显著的序列模式（前10）:")
    sequential_patterns = result['sequential_patterns']
    pattern_list = []
    total_patterns = sum(sum(followers.values()) for followers in sequential_patterns.values())
    for cat1, followers in sequential_patterns.items():
        for cat2, count in followers.items():
            pattern_percentage = (count / total_patterns * 100)
            pattern_list.append((cat1, cat2, pattern_percentage))
    top_patterns = sorted(pattern_list, key=lambda x: x[2], reverse=True)[:10]
    print("\n购买序列模式及其占比(%):")
    for cat1, cat2, percentage in top_patterns:
        print(f"{cat1} -> {cat2}: {percentage:.2f}%")

def visualize_time_analysis(result, save_path=None):
    plt.figure(figsize=(15, 10))
    seasonal_df = pd.DataFrame(result['seasonal_patterns']).fillna(0)
    seasonal_df = (seasonal_df / seasonal_df.sum().sum() * 100).round(2)
    plt.subplot(2, 2, 1)
    sns.heatmap(seasonal_df, annot=True, fmt='.1f', cmap='YlOrRd')
    plt.title('季节性购买模式热力图')
    plt.subplot(2, 2, 2)
    monthly_df = pd.DataFrame(result['monthly_patterns']).fillna(0)
    monthly_total = monthly_df.sum().sum()
    monthly_proportions = (monthly_df.sum() / monthly_total * 100).round(2)
    monthly_proportions.plot(kind='bar')
    plt.title('月度购买频率分布')
    plt.xlabel('月份')
    plt.ylabel('占比 (%)')
    plt.xticks(rotation=45)
    plt.subplot(2, 2, 3)
    weekly_df = pd.DataFrame(result['weekly_patterns']).fillna(0)
    weekly_total = weekly_df.sum().sum()
    weekly_proportions = (weekly_df.sum() / weekly_total * 100).round(2)
    weekly_proportions.plot(kind='bar')
    plt.title('周度购买频率分布')
    plt.xlabel('星期')
    plt.ylabel('占比 (%)')
    plt.xticks(rotation=45)
    plt.subplot(2, 2, 4)
    category_features = result['category_time_features']
    purchase_counts = {cat: feat['total_purchases'] for cat, feat in category_features.items()}
    pd.Series(purchase_counts).nlargest(10).plot(kind='bar')
    plt.title('Top10类别购买频次')
    plt.xlabel('类别')
    plt.ylabel('购买次数')
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}/time_analysis.png")
    plt.show()

def save_analysis_result(result, save_dir='./results', prefix='time'):
    os.makedirs(save_dir, exist_ok=True)
    # 保存所有分析结果为json
    with open(os.path.join(save_dir, f'{prefix}_analysis.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)

def main3():
    os.makedirs('./results', exist_ok=True)
    params = {
        'parquet_dir': 'D:\\mylearn\\data_mining\\new_10G_data',
        'product_catalog_path': 'product_catalog.json/product_catalog.json',
        'batch_size': 1000000,
        'columns': ['purchase_history'],
        'sample_ratio': 1
    }
    print("=== 3.开始时间序列模式挖掘分析 ===")
    print(f"数据目录: {params['parquet_dir']}")
    print(f"商品目录: {params['product_catalog_path']}")
    result = analyze_time_patterns(**params)
    print_time_analysis(result)
    print("\n4. 生成可视化结果...")
    visualize_time_analysis(result, save_path='./results')
    print("\n5. 保存分析结果...")
    save_analysis_result(result, save_dir='./results', prefix='3_time')

if __name__ == "__main__":
    main3() 