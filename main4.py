import pandas as pd
import json
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

def extract_refund_categories(purchase_history, id_to_info):
    try:
        if isinstance(purchase_history, str):
            purchase_data = json.loads(purchase_history)
        else:
            purchase_data = purchase_history
        payment_status = purchase_data.get('payment_status', '')
        is_refund = payment_status in ['已退款', '部分退款']
        categories = set()
        for item in purchase_data.get('items', []):
            item_id = str(item.get('id'))
            if item_id in id_to_info:
                sub_category = id_to_info[item_id]['category']
                main_category = get_main_category(sub_category)
                categories.add(main_category)
        return is_refund, categories
    except Exception as e:
        print(f"解析购买历史时出错: {str(e)}")
        return False, set()

def analyze_refund_patterns(
    parquet_dir,
    product_catalog_path,
    min_support=0.005,
    min_confidence=0.4,
    batch_size=1000000,
    max_categories_per_batch=1000,
    columns=None,
    sample_ratio=1.0
):
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder
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
        category_refund_stats = defaultdict(lambda: {'refund': 0, 'total': 0})
        total_orders = len(df)
        refund_orders = 0
        all_refund_categories = []
        print("\n第一阶段：收集退款数据...")
        for purchase_history in tqdm(df['purchase_history'], desc="处理退款数据", ncols=100):
            is_refund, categories = extract_refund_categories(purchase_history, id_to_info)
            if categories:
                for category in categories:
                    category_refund_stats[category]['total'] += 1
                    if is_refund:
                        category_refund_stats[category]['refund'] += 1
                if is_refund:
                    refund_orders += 1
                    all_refund_categories.append(list(categories))
        print("\n第二阶段：分析退款类别...")
        category_refund_rates = {}
        for category, stats in category_refund_stats.items():
            if stats['total'] > 0:
                refund_rate = (stats['refund'] / stats['total'] * 100)
                category_refund_rates[category] = {
                    'refund_count': stats['refund'],
                    'total_count': stats['total'],
                    'refund_rate': refund_rate
                }
        sorted_categories = sorted(
            category_refund_rates.items(),
            key=lambda x: x[1]['refund_count'],
            reverse=True
        )
        top_categories = [cat for cat, _ in sorted_categories[:max_categories_per_batch]]
        print(f"\n第三阶段：对top {len(top_categories)} 个类别进行关联规则分析...")
        filtered_refund_categories = []
        for categories in all_refund_categories:
            filtered_cats = [cat for cat in categories if cat in top_categories]
            if len(filtered_cats) >= 2:
                filtered_refund_categories.append(filtered_cats)
        print("\n开始关联规则挖掘...")
        te = TransactionEncoder()
        te_ary = te.fit(filtered_refund_categories).transform(filtered_refund_categories)
        df_te = pd.DataFrame(te_ary, columns=te.columns_)
        frequent_itemsets = apriori(df_te, min_support=min_support, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        return {
            'rules': rules,
            'category_refund_rates': category_refund_rates,
            'stats': {
                'total_orders': total_orders,
                'refund_orders': refund_orders,
                'refund_rate': (refund_orders / total_orders * 100),
                'total_rules': len(rules),
                'analyzed_categories': len(top_categories)
            }
        }
    except Exception as e:
        raise Exception(f"退款模式分析过程中出错: {str(e)}")

def print_refund_analysis(result):
    print("\n=== 退款模式分析结果 ===")
    print("\n1. 基本统计信息:")
    stats = result['stats']
    print(f"总订单数: {stats['total_orders']}")
    print(f"退款订单数: {stats['refund_orders']}")
    print(f"总体退款率: {stats['refund_rate']:.2f}%")
    print(f"发现规则数: {stats['total_rules']}")
    print("\n2. 各类别退款情况:")
    refund_rates_df = pd.DataFrame.from_dict(result['category_refund_rates'], orient='index')
    sorted_rates = refund_rates_df.sort_values('refund_rate', ascending=False)
    for category, row in sorted_rates.iterrows():
        print(f"\n{category}:")
        print(f"  退款率: {row['refund_rate']:.2f}%")
        print(f"  退款订单数: {row['refund_count']}")
        print(f"  总订单数: {row['total_count']}")
    print("\n3. 退款商品组合规则:")
    rules_df = result['rules']
    if len(rules_df) > 0:
        print("\n发现的关联规则（按置信度排序）:")
        sorted_rules = rules_df.sort_values('confidence', ascending=False)
        for idx, rule in sorted_rules.iterrows():
            print(f"\n规则 {idx + 1}:")
            print(f"前项: {list(rule['antecedents'])}")
            print(f"后项: {list(rule['consequents'])}")
            print(f"支持度: {rule['support']:.3f}")
            print(f"置信度: {rule['confidence']:.3f}")
            print(f"提升度: {rule['lift']:.3f}")
    else:
        print("没有找到满足条件的关联规则")

def visualize_refund_analysis(result, save_path=None):
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    refund_rates = pd.DataFrame.from_dict(result['category_refund_rates'], orient='index')
    sns.histplot(data=refund_rates, x='refund_rate', bins=20)
    plt.title('类别退款率分布')
    plt.xlabel('退款率 (%)')
    plt.ylabel('类别数量')
    plt.subplot(2, 2, 2)
    top_refund_rates = refund_rates.nlargest(10, 'refund_rate')
    top_refund_rates['refund_rate'].plot(kind='bar')
    plt.title('Top10高退款率类别')
    plt.xlabel('类别')
    plt.ylabel('退款率 (%)')
    plt.xticks(rotation=45)
    plt.subplot(2, 2, 3)
    rules_df = result['rules']
    if len(rules_df) > 0:
        sns.scatterplot(data=rules_df, x='support', y='confidence', size='lift', sizes=(50, 400), alpha=0.6)
        plt.title('退款关联规则分析')
        plt.xlabel('支持度')
        plt.ylabel('置信度')
    plt.subplot(2, 2, 4)
    stats = result['stats']
    plt.pie([stats['refund_orders'], stats['total_orders'] - stats['refund_orders']],
            labels=['退款订单', '正常订单'],
            autopct='%1.1f%%')
    plt.title('退款订单占比')
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}/refund_analysis.png")
    plt.show()

def save_analysis_result(result, save_dir='./results', prefix='refund'):
    os.makedirs(save_dir, exist_ok=True)
    # 保存所有规则
    result['rules'].to_csv(os.path.join(save_dir, f'{prefix}_rules.csv'), index=False)
    # 保存退款率
    pd.DataFrame.from_dict(result['category_refund_rates'], orient='index').to_csv(os.path.join(save_dir, f'{prefix}_refund_rates.csv'))
    # 保存统计信息
    with open(os.path.join(save_dir, f'{prefix}_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(result['stats'], f, ensure_ascii=False, indent=2)

def main4():
    os.makedirs('./results', exist_ok=True)
    params = {
        'parquet_dir': 'D:\\mylearn\\data_mining\\new_10G_data',
        'product_catalog_path': 'product_catalog.json/product_catalog.json',
        'min_support': 0.005,
        'min_confidence': 0.4,
        'batch_size': 1000000,
        'max_categories_per_batch': 100,
        'columns': ['purchase_history'],
        'sample_ratio': 1
    }
    print("=== 4.开始退款模式分析 ===")
    print(f"数据目录: {params['parquet_dir']}")
    print(f"商品目录: {params['product_catalog_path']}")
    result = analyze_refund_patterns(**params)
    print_refund_analysis(result)
    print("\n4. 生成可视化结果...")
    visualize_refund_analysis(result, save_path='./results')
    print("\n5. 保存分析结果...")
    save_analysis_result(result, save_dir='./results', prefix='4_refund')

if __name__ == "__main__":
    main4()
