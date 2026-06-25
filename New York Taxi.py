import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import warnings

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("第一阶段：数据加载与初步探索 (EDA)")
print("=" * 60)

# ============================================================
# Step 1: 设置数据路径
# ============================================================
DATA_DIR = './'

# 查看data目录下有哪些文件
print(f"\n检查数据目录: {os.path.abspath(DATA_DIR)}")
print("目录下的文件:")
for f in os.listdir(DATA_DIR):
    if f.endswith('.parquet'):
        print(f"  - {f}")

# ============================================================
# Step 2: 加载数据
# ============================================================
# 读取所有parquet文件
parquet_files = glob.glob(os.path.join(DATA_DIR, '*.parquet'))
print(f"\n找到 {len(parquet_files)} 个parquet文件")

if len(parquet_files) == 0:
    print("\n⚠️ 没有找到parquet文件！")
    print("请确保已将下载的 .parquet 文件放在 data/ 目录下")
    print("当前目录下的文件列表:")
    for f in os.listdir(DATA_DIR):
        print(f"  - {f}")
else:
    # 加载所有parquet文件并合并
    df_list = []
    for file in parquet_files:
        print(f"正在加载: {os.path.basename(file)}")
        df_temp = pd.read_parquet(file)
        df_list.append(df_temp)
        print(f"  行数: {len(df_temp):,}")

    df = pd.concat(df_list, ignore_index=True)
    print(f"\n✅ 合并后总数据量: {len(df):,} 行")
    print(f"✅ 总列数: {len(df.columns)} 列")

    # ============================================================
    # Step 3: 查看数据结构
    # ============================================================
    print("\n" + "=" * 60)
    print("数据结构概览")
    print("=" * 60)
    print("\n列名及数据类型:")
    print(df.dtypes)

    print("\n前5行数据:")
    print(df.head())

    print("\n数值型字段统计:")
    print(df.describe())

    print("\n缺失值统计:")
    missing = df.isnull().sum()
    print(missing[missing > 0] if any(missing > 0) else "无缺失值")

    # ============================================================
    # Step 4: 计算目标变量 (行程时长)
    # ============================================================
    print("\n" + "=" * 60)
    print("计算目标变量：行程时长 (trip_duration)")
    print("=" * 60)

    # 转换为datetime格式
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])

    # 计算时长（秒）
    df['trip_duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds()

    print(f"\n行程时长统计:")
    print(df['trip_duration'].describe())
    print(f"\n行程时长为负的记录数: {(df['trip_duration'] < 0).sum()}")
    print(f"行程时长为0的记录数: {(df['trip_duration'] == 0).sum()}")

    # ============================================================
    # Step 5: 可视化目标变量分布
    # ============================================================
    print("\n" + "=" * 60)
    print("绘制目标变量分布图")
    print("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：原始分布（去掉过长行程便于观察主体）
    duration_clean = df[df['trip_duration'] < 10000]['trip_duration']
    sns.histplot(duration_clean, bins=100, ax=axes[0])
    axes[0].set_title('行程时长分布（原始，<10000秒）')
    axes[0].set_xlabel('时长（秒）')

    # 右图：取对数后的分布
    sns.histplot(np.log1p(df['trip_duration']), bins=100, ax=axes[1])
    axes[1].set_title('行程时长分布（取对数后）')
    axes[1].set_xlabel('log(时长)')

    plt.tight_layout()
    plt.savefig('eda_duration_distribution.png', dpi=150)
    print("✅ 图片已保存为: eda_duration_distribution.png")
    plt.show()

    # ============================================================
    # Step 6: 查看关键字段分布
    # ============================================================
    print("\n" + "=" * 60)
    print("关键字段分布")
    print("=" * 60)

    print("\n乘客数分布:")
    print(df['passenger_count'].value_counts().sort_index())

    print("\n行程距离统计:")
    print(df['trip_distance'].describe())

    print("\n行程距离为0的记录数: {(df['trip_distance'] == 0).sum()}")

    # ============================================================
    # Step 7: 距离vs时长散点图
    # ============================================================
    print("\n" + "=" * 60)
    print("绘制 距离 vs 时长 散点图")
    print("=" * 60)

    plt.figure(figsize=(10, 6))
    # 采样一部分数据绘制，避免点太多卡顿
    sample_df = df.sample(min(50000, len(df)), random_state=42)
    plt.scatter(sample_df['trip_distance'], sample_df['trip_duration'], alpha=0.1, s=1)
    plt.xlabel('行程距离（英里）')
    plt.ylabel('行程时长（秒）')
    plt.title('行程距离 vs 行程时长')
    plt.xlim(0, 30)
    plt.ylim(0, 10000)
    plt.savefig('eda_distance_vs_duration.png', dpi=150)
    print("✅ 图片已保存为: eda_distance_vs_duration.png")
    plt.show()

    print("\n✅ 第一阶段完成！")
    print("\n" + "=" * 60)
    print("第二阶段：数据清洗")
    print("=" * 60)

    # 复制一份数据，避免修改原始数据
    df_clean = df.copy()

    print(f"\n清洗前数据量: {len(df_clean):,} 行")

    # ============================================================
    # 1. 删除行程时长为负或为0的记录
    # ============================================================
    print("\n" + "-" * 40)
    print("1. 处理行程时长异常")
    print("-" * 40)

    invalid_duration = (df_clean['trip_duration'] <= 0).sum()
    df_clean = df_clean[df_clean['trip_duration'] > 0]
    print(f"删除时长≤0的记录: {invalid_duration:,} 条")
    print(f"删除后剩余: {len(df_clean):,} 行")

    # ============================================================
    # 2. 删除行程距离为0或过短的记录
    # ============================================================
    print("\n" + "-" * 40)
    print("2. 处理行程距离异常")
    print("-" * 40)

    invalid_distance = (df_clean['trip_distance'] <= 0).sum()
    df_clean = df_clean[df_clean['trip_distance'] > 0]
    print(f"删除距离≤0的记录: {invalid_distance:,} 条")

    # 删除距离过长的异常值（99.9分位数以上）
    distance_999 = df_clean['trip_distance'].quantile(0.999)
    long_distance = (df_clean['trip_distance'] > distance_999).sum()
    df_clean = df_clean[df_clean['trip_distance'] <= distance_999]
    print(f"删除距离>{distance_999:.1f}英里的记录: {long_distance:,} 条")
    print(f"删除后剩余: {len(df_clean):,} 行")

    # ============================================================
    # 3. 删除行程时长过长的异常值
    # ============================================================
    print("\n" + "-" * 40)
    print("3. 处理行程时长过长")
    print("-" * 40)

    duration_999 = df_clean['trip_duration'].quantile(0.999)
    long_duration = (df_clean['trip_duration'] > duration_999).sum()
    df_clean = df_clean[df_clean['trip_duration'] <= duration_999]
    print(f"删除时长>{duration_999:.0f}秒的记录: {long_duration:,} 条")
    print(f"删除后剩余: {len(df_clean):,} 行")

    # ============================================================
    # 4. 处理乘客数异常
    # ============================================================
    print("\n" + "-" * 40)
    print("4. 处理乘客数异常")
    print("-" * 40)

    print("原始乘客数分布:")
    print(df_clean['passenger_count'].value_counts().sort_index())

    # 乘客数必须为1-6之间的整数
    invalid_passenger = ((df_clean['passenger_count'] < 1) | (df_clean['passenger_count'] > 6)).sum()
    df_clean = df_clean[(df_clean['passenger_count'] >= 1) & (df_clean['passenger_count'] <= 6)]
    print(f"删除乘客数<1或>6的记录: {invalid_passenger:,} 条")
    print(f"删除后剩余: {len(df_clean):,} 行")

    print("\n清洗后乘客数分布:")
    print(df_clean['passenger_count'].value_counts().sort_index())

    # ============================================================
    # 5. 检查是否有缺失值
    # ============================================================
    print("\n" + "-" * 40)
    print("5. 检查缺失值")
    print("-" * 40)

    missing = df_clean.isnull().sum()
    if missing.sum() > 0:
        print("存在缺失值的列:")
        print(missing[missing > 0])
    else:
        print("✅ 无缺失值")

    # ============================================================
    # 6. 查看清洗后的数据统计
    # ============================================================
    print("\n" + "-" * 40)
    print("6. 清洗后数据概览")
    print("-" * 40)

    print(f"\n✅ 最终数据量: {len(df_clean):,} 行")
    print(f"✅ 最终列数: {len(df_clean.columns)} 列")

    print("\n清洗后目标变量统计:")
    print(df_clean['trip_duration'].describe())

    print("\n清洗后距离统计:")
    print(df_clean['trip_distance'].describe())

    # ============================================================
    # 7. 可视化清洗前后的对比
    # ============================================================
    print("\n" + "-" * 40)
    print("7. 绘制清洗前后对比图")
    print("-" * 40)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 清洗前
    sns.histplot(df[df['trip_duration'] < 10000]['trip_duration'], bins=80, ax=axes[0], color='red', alpha=0.6)
    axes[0].set_title('清洗前（<10000秒）')
    axes[0].set_xlabel('时长（秒）')

    # 清洗后
    sns.histplot(df_clean[df_clean['trip_duration'] < 10000]['trip_duration'], bins=80, ax=axes[1], color='blue',
                 alpha=0.6)
    axes[1].set_title('清洗后（<10000秒）')
    axes[1].set_xlabel('时长（秒）')

    plt.tight_layout()
    plt.savefig('data_cleaning_comparison.png', dpi=150)
    print("✅ 图片已保存为: data_cleaning_comparison.png")
    plt.show()

    print("\n" + "=" * 60)
    print("✅ 第二阶段：数据清洗完成！")
    print("=" * 60)
    print("\n" + "=" * 60)
    print("第三阶段：特征工程")
    print("=" * 60)

    # ============================================================
    # 1. 时间特征提取
    # ============================================================
    print("\n" + "-" * 40)
    print("1. 提取时间特征")
    print("-" * 40)

    # 从pickup时间中提取各种时间特征
    df_clean['pickup_hour'] = df_clean['tpep_pickup_datetime'].dt.hour
    df_clean['pickup_day'] = df_clean['tpep_pickup_datetime'].dt.day
    df_clean['pickup_month'] = df_clean['tpep_pickup_datetime'].dt.month
    df_clean['pickup_dayofweek'] = df_clean['tpep_pickup_datetime'].dt.dayofweek  # 0=周一, 6=周日
    df_clean['pickup_quarter'] = df_clean['tpep_pickup_datetime'].dt.quarter

    # 是否为周末 (周六=5, 周日=6)
    df_clean['is_weekend'] = df_clean['pickup_dayofweek'].apply(lambda x: 1 if x >= 5 else 0)


    # 是否为早晚高峰 (工作日：7-9点, 16-19点)
    def is_rush_hour(hour, dayofweek):
        if dayofweek >= 5:  # 周末
            return 0
        if (hour >= 7 and hour <= 9) or (hour >= 16 and hour <= 19):
            return 1
        return 0


    df_clean['is_rush_hour'] = df_clean.apply(
        lambda row: is_rush_hour(row['pickup_hour'], row['pickup_dayofweek']),
        axis=1
    )

    # 是否为深夜时段 (23-5点)
    df_clean['is_night'] = df_clean['pickup_hour'].apply(lambda x: 1 if x >= 23 or x <= 5 else 0)

    print("✅ 时间特征提取完成")
    print(f"   - pickup_hour: 0-23")
    print(f"   - pickup_dayofweek: 0-6")
    print(f"   - is_weekend: 0/1")
    print(f"   - is_rush_hour: 0/1")
    print(f"   - is_night: 0/1")

    # ============================================================
    # 2. 计算速度特征 (距离/时长)
    # ============================================================
    print("\n" + "-" * 40)
    print("2. 计算速度特征")
    print("-" * 40)

    # 速度 = 距离 / 时长 (英里/秒)
    df_clean['speed_mps'] = df_clean['trip_distance'] / df_clean['trip_duration']
    # 转换为英里/小时
    df_clean['speed_mph'] = df_clean['speed_mps'] * 3600

    # 处理速度异常值 (速度 > 100 mph 或 < 0.5 mph 可能异常)
    print(f"速度统计 (mph):")
    print(df_clean['speed_mph'].describe())

    # 限制速度范围，避免极端值影响模型
    df_clean['speed_mph'] = df_clean['speed_mph'].clip(0.5, 100)

    print("✅ 速度特征计算完成")

    # ============================================================
    # 3. 尝试加载 Taxi Zones 数据 (如果已下载)
    # ============================================================
    print("\n" + "-" * 40)
    print("3. 加载区域查表数据")
    print("-" * 40)

    try:
        # 尝试读取taxi zone lookup文件
        import os

        zone_file = None
        for f in os.listdir('.'):
            if 'taxi_zone' in f.lower() and f.endswith('.csv'):
                zone_file = f
                break
        if zone_file:
            taxi_zones = pd.read_csv(zone_file)
            print(f"✅ 找到区域数据: {zone_file}")
            print(f"   区域数量: {len(taxi_zones)}")
            print(taxi_zones.head())
        else:
            print("⚠️ 未找到 taxi_zone_lookup.csv 文件")
            print("   暂时使用区域ID作为特征，后续可补充")
    except Exception as e:
        print(f"⚠️ 加载区域数据失败: {e}")
        print("   暂时使用区域ID作为特征")

    # ============================================================
    # 4. 查看特征工程后的数据
    # ============================================================
    print("\n" + "-" * 40)
    print("4. 特征工程后数据概览")
    print("-" * 40)

    print(f"\n当前总列数: {len(df_clean.columns)}")
    print("\n新增特征:")
    new_features = ['pickup_hour', 'pickup_day', 'pickup_month', 'pickup_dayofweek',
                    'pickup_quarter', 'is_weekend', 'is_rush_hour', 'is_night',
                    'speed_mps', 'speed_mph']
    for f in new_features:
        if f in df_clean.columns:
            print(f"  - {f}")

    # ============================================================
    # 5. 可视化新增特征
    # ============================================================
    print("\n" + "-" * 40)
    print("5. 绘制新增特征分布图")
    print("-" * 40)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 按小时统计行程数量
    hourly_counts = df_clean['pickup_hour'].value_counts().sort_index()
    axes[0, 0].bar(hourly_counts.index, hourly_counts.values)
    axes[0, 0].set_title('各小时叫车数量')
    axes[0, 0].set_xlabel('小时')
    axes[0, 0].set_ylabel('数量')

    # 按星期统计行程数量
    dow_counts = df_clean['pickup_dayofweek'].value_counts().sort_index()
    axes[0, 1].bar(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], dow_counts.values)
    axes[0, 1].set_title('各星期叫车数量')

    # 速度分布
    sns.histplot(df_clean['speed_mph'], bins=50, ax=axes[0, 2])
    axes[0, 2].set_title('速度分布 (mph)')
    axes[0, 2].set_xlim(0, 60)

    # 高峰vs非高峰
    rush_counts = df_clean['is_rush_hour'].value_counts()
    axes[1, 0].bar(['非高峰', '高峰'], rush_counts.values)
    axes[1, 0].set_title('高峰时段叫车数量对比')

    # 周末vs工作日
    weekend_counts = df_clean['is_weekend'].value_counts()
    axes[1, 1].bar(['工作日', '周末'], weekend_counts.values)
    axes[1, 1].set_title('周末vs工作日叫车数量对比')

    # 深夜vs其他
    night_counts = df_clean['is_night'].value_counts()
    axes[1, 2].bar(['非深夜', '深夜'], night_counts.values)
    axes[1, 2].set_title('深夜时段叫车数量对比')

    plt.tight_layout()
    plt.savefig('feature_engineering_distributions.png', dpi=150)
    print("✅ 图片已保存为: feature_engineering_distributions.png")
    plt.show()

    print("\n" + "=" * 60)
    print("✅ 第三阶段：特征工程完成！")
    print("=" * 60)

    print("\n📌 新增特征汇总:")
    print("  - 时间特征: pickup_hour, pickup_day, pickup_month, pickup_dayofweek")
    print("  - 周期特征: is_weekend, is_rush_hour, is_night")
    print("  - 速度特征: speed_mph")
    print("  - 区域特征: PULocationID, DOLocationID (待与Taxi Zones关联)")
# 保存数据
df_clean.to_pickle('df_clean.pkl')
print("✅ 数据已保存为 df_clean.pkl")