import pandas as pd
import json
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import glob
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
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

def extract_payment_and_categories(purchase_history, id_to_info, price_threshold=5000):
    try:
        if isinstance(purchase_history, str):
            purchase_data = json.loads(purchase_history)
        else:
            purchase_data = purchase_history
        payment_method = purchase_data.get('payment_method', '未知')
        categories = set()
        has_high_value = False
        for item in purchase_data.get('items', []):
            item_id = str(item.get('id'))
            if item_id in id_to_info:
                sub_category = id_to_info[item_id]['category']
                main_category = get_main_category(sub_category)
                categories.add(main_category)
                if id_to_info[item_id]['price'] > price_threshold:
                    has_high_value = True
        return payment_method, categories, has_high_value
    except Exception as e:
        print(f"解析购买历史时出错: {str(e)}")
        return None, set(), False

def analyze_payment_category_association(
    parquet_dir,
    product_catalog_path,
    min_support=0.01,
    min_confidence=0.6,
    price_threshold=5000,
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
        payment_categories = []
        high_value_payments = []
        for purchase_history in tqdm(df['purchase_history'], desc="处理订单数据", ncols=100):
            payment_method, categories, has_high_value = extract_payment_and_categories(
                purchase_history, id_to_info, price_threshold
            )
            if payment_method and categories:
                all_items = {f"支付方式_{payment_method}"} | {f"类别_{cat}" for cat in categories}
                payment_categories.append(list(all_items))
                if has_high_value:
                    high_value_payments.append(payment_method)
        te = TransactionEncoder()
        te_ary = te.fit(payment_categories).transform(payment_categories)
        df_te = pd.DataFrame(te_ary, columns=te.columns_)
        frequent_itemsets = apriori(df_te, min_support=min_support, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        high_value_payment_stats = pd.Series(high_value_payments).value_counts()
        total_high_value = len(high_value_payments)
        high_value_payment_pct = (high_value_payment_stats / total_high_value * 100).round(2)
        return {
            'rules': rules,
            'high_value_payments': high_value_payment_pct,
            'stats': {
                'total_transactions': len(payment_categories),
                'high_value_transactions': total_high_value,
                'total_rules': len(rules)
            }
        }
    except Exception as e:
        raise Exception(f"分析过程中出错: {str(e)}")

def print_payment_analysis(result):
    print("\n=== 支付方式与商品类别关联分析结果 ===")
    print("\n1. 基本统计信息:")
    for key, value in result['stats'].items():
        print(f"{key}: {value}")
    print("\n2. 高价值商品(>5000元)的支付方式分布:")
    print(result['high_value_payments'])
    print("\n3. 支付方式与商品类别的关联规则:")
    rules_df = result['rules']
    if len(rules_df) > 0:
        payment_rules = rules_df[
            rules_df['antecedents'].apply(lambda x: any('支付方式_' in str(item) for item in x)) |
            rules_df['consequents'].apply(lambda x: any('支付方式_' in str(item) for item in x))
        ]
        print("\n支付方式相关的规则:")
        print(payment_rules.sort_values('confidence', ascending=False))
    else:
        print("没有找到满足条件的关联规则")

def visualize_payment_analysis(result, save_path=None):
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    result['high_value_payments'].plot(kind='bar')
    plt.title('高价值商品支付方式分布')
    plt.xlabel('支付方式')
    plt.ylabel('占比 (%)')
    plt.xticks(rotation=45)
    plt.subplot(2, 1, 2)
    rules_df = result['rules']
    if len(rules_df) > 0:
        top_rules = rules_df.nlargest(10, 'lift')
        sns.scatterplot(data=top_rules, x='support', y='confidence', size='lift', sizes=(50, 400), alpha=0.6)
        plt.title('支付方式关联规则Top10')
        plt.xlabel('支持度')
        plt.ylabel('置信度')
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}/payment_analysis.png")
    plt.show()

def save_analysis_result(result, save_dir='./results', prefix='payment'):
    os.makedirs(save_dir, exist_ok=True)
    # 保存所有规则
    result['rules'].to_csv(os.path.join(save_dir, f'{prefix}_rules.csv'), index=False)
    # 保存高价值商品支付方式分布
    result['high_value_payments'].to_csv(os.path.join(save_dir, f'{prefix}_high_value_payments.csv'))
    # 保存统计信息
    with open(os.path.join(save_dir, f'{prefix}_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(result['stats'], f, ensure_ascii=False, indent=2)

def main2():
    os.makedirs('./results', exist_ok=True)
    params = {
        'parquet_dir': 'D:\\mylearn\\data_mining\\new_10G_data',
        'product_catalog_path': 'product_catalog.json/product_catalog.json',
        'min_support': 0.005,
        'min_confidence': 0.4,
        'price_threshold': 5000,
        'batch_size': 1000000,
        'columns': ['purchase_history'],
        'sample_ratio': 0.2
    }
    print("=== 2.开始支付方式与商品类别关联分析 ===")
    print(f"数据目录: {params['parquet_dir']}")
    print(f"商品目录: {params['product_catalog_path']}")
    result = analyze_payment_category_association(**params)
    print_payment_analysis(result)
    print("\n4. 生成可视化结果...")
    visualize_payment_analysis(result, save_path='./results')
    print("\n5. 保存分析结果...")
    save_analysis_result(result, save_dir='./results', prefix='2_payment')

if __name__ == "__main__":
    main2() 