"""
纽约市出租车行程时长预测 - 模型训练与评估
数据科学/大数据专业 - 实习项目
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
import warnings
import pickle
import os

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 70)
print("第四阶段：模型训练与评估")
print("=" * 70)

# ============================================================
# Step 1: 加载清洗好的数据
# ============================================================
print("\n" + "-" * 50)
print("1. 加载数据")
print("-" * 50)

# 检查 df_clean 是否存在
try:
    if 'df_clean' not in dir():
        if os.path.exists('df_clean.pkl'):
            df_clean = pd.read_pickle('df_clean.pkl')
            print("✅ 从 df_clean.pkl 加载数据")
        else:
            print("❌ 未找到数据文件，请先运行特征工程代码")
            exit()
    else:
        print("✅ 使用当前内存中的 df_clean 数据")
except NameError:
    print("❌ 错误: df_clean 未定义")
    exit()

print(f"数据量: {len(df_clean):,} 行")
print(f"列数: {len(df_clean.columns)} 列")

# 显示所有列名，检查是否有datetime类型
print("\n所有列名及类型:")
for col in df_clean.columns:
    print(f"  {col}: {df_clean[col].dtype}")

# ============================================================
# Step 2: 准备特征和目标变量
# ============================================================
print("\n" + "-" * 50)
print("2. 准备特征和目标变量")
print("-" * 50)

# 目标变量
target = 'trip_duration'
y = np.log1p(df_clean[target])

# ============================================================
# 关键：排除所有非数值类型的列
# ============================================================
exclude_cols = [
    target,
    'tpep_pickup_datetime',   # datetime类型，必须排除
    'tpep_dropoff_datetime',  # datetime类型，必须排除
]

# 自动排除所有非数值类型列
for col in df_clean.columns:
    if col not in exclude_cols:
        # 如果是object、datetime或category类型，加入排除列表
        if df_clean[col].dtype == 'object' or 'datetime' in str(df_clean[col].dtype):
            exclude_cols.append(col)
        # 如果是category类型，需要转换为数值
        elif df_clean[col].dtype.name == 'category':
            df_clean[col] = df_clean[col].astype(str)
            le = LabelEncoder()
            df_clean[col] = le.fit_transform(df_clean[col])
            print(f"  转换 category 列: {col}")

# 保留的列
feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
print(f"\n原始列数: {len(df_clean.columns)}")
print(f"排除列数: {len(exclude_cols)}")
print(f"保留特征数: {len(feature_cols)}")

# 打印保留的特征
print("\n保留的特征列:")
for i, col in enumerate(feature_cols):
    print(f"  {i+1}. {col} ({df_clean[col].dtype})")

# ============================================================
# Step 3: 处理类别特征
# ============================================================
print("\n" + "-" * 50)
print("3. 处理类别特征")
print("-" * 50)

# 找出需要Label Encoding的类别特征
categorical_cols = []
for col in feature_cols:
    if df_clean[col].dtype == 'object':
        categorical_cols.append(col)
    elif df_clean[col].dtype.name == 'category':
        categorical_cols.append(col)

print(f"需要编码的类别特征: {categorical_cols}")

# 对类别特征进行Label Encoding
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_clean[col] = df_clean[col].astype(str)
    df_clean[col] = le.fit_transform(df_clean[col])
    label_encoders[col] = le
    print(f"  - {col}: {len(le.classes_)} 个类别")

# 准备特征矩阵
X = df_clean[feature_cols]
print(f"\n✅ 特征矩阵形状: {X.shape}")
print(f"✅ 特征矩阵数据类型:")
print(X.dtypes.value_counts())

# ============================================================
# Step 4: 划分训练集和验证集
# ============================================================
print("\n" + "-" * 50)
print("4. 划分训练集和验证集")
print("-" * 50)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"训练集: {len(X_train):,} 行")
print(f"验证集: {len(X_val):,} 行")

# ============================================================
# Step 5: LightGBM 模型训练
# ============================================================
print("\n" + "-" * 50)
print("5. LightGBM 模型训练")
print("-" * 50)

# 创建LightGBM数据集
lgb_train = lgb.Dataset(X_train, y_train)
lgb_valid = lgb.Dataset(X_val, y_val, reference=lgb_train)

# 设置参数
lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'n_jobs': -1,
    'random_state': 42
}

print("开始训练 LightGBM...")
lgb_model = lgb.train(
    lgb_params,
    lgb_train,
    num_boost_round=500,
    valid_sets=[lgb_train, lgb_valid],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
)

# 验证集预测
y_pred_lgb = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)

# 计算RMSLE
def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(np.exp(y_true) - 1, np.exp(y_pred) - 1))

lgb_rmsle = rmsle(y_val, y_pred_lgb)
print(f"\n✅ LightGBM 验证集 RMSLE: {lgb_rmsle:.5f}")

# ============================================================
# Step 6: XGBoost 模型训练
# ============================================================
print("\n" + "-" * 50)
print("6. XGBoost 模型训练")
print("-" * 50)

xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_estimators': 500,
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': 0
}

print("开始训练 XGBoost...")
xgb_model = xgb.XGBRegressor(**xgb_params)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

# 验证集预测
y_pred_xgb = xgb_model.predict(X_val)
xgb_rmsle = rmsle(y_val, y_pred_xgb)
print(f"\n✅ XGBoost 验证集 RMSLE: {xgb_rmsle:.5f}")

# ============================================================
# Step 7: 模型融合
# ============================================================
print("\n" + "-" * 50)
print("7. 模型融合（加权平均）")
print("-" * 50)

best_weight = 0.5
best_ensemble_rmsle = float('inf')

for weight in np.arange(0, 1.1, 0.1):
    y_pred_ensemble = weight * y_pred_lgb + (1 - weight) * y_pred_xgb
    ensemble_rmsle = rmsle(y_val, y_pred_ensemble)
    if ensemble_rmsle < best_ensemble_rmsle:
        best_ensemble_rmsle = ensemble_rmsle
        best_weight = weight

print(f"最优 LightGBM 权重: {best_weight:.1f}")
print(f"最优 XGBoost 权重: {1 - best_weight:.1f}")
print(f"✅ 模型融合验证集 RMSLE: {best_ensemble_rmsle:.5f}")

# ============================================================
# Step 8: 特征重要性分析
# ============================================================
print("\n" + "-" * 50)
print("8. 特征重要性分析")
print("-" * 50)

# LightGBM 特征重要性
lgb_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': lgb_model.feature_importance()
}).sort_values('importance', ascending=False)

print("\nTop 10 特征 (LightGBM):")
print(lgb_importance.head(10))

# 可视化特征重要性
fig, axes = plt.subplots(1, 2, figsize=(15, 8))

# LightGBM
top_features_lgb = lgb_importance.head(15)
axes[0].barh(top_features_lgb['feature'], top_features_lgb['importance'])
axes[0].set_title('LightGBM 特征重要性 (Top 15)')
axes[0].set_xlabel('重要性')
axes[0].invert_yaxis()

# XGBoost 特征重要性
xgb_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

top_features_xgb = xgb_importance.head(15)
axes[1].barh(top_features_xgb['feature'], top_features_xgb['importance'])
axes[1].set_title('XGBoost 特征重要性 (Top 15)')
axes[1].set_xlabel('重要性')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
print("\n✅ 图片已保存为: feature_importance.png")
plt.show()

# ============================================================
# Step 9: 保存模型
# ============================================================
print("\n" + "-" * 50)
print("9. 保存模型")
print("-" * 50)

lgb_model.save_model('lgb_model.txt')
print("✅ LightGBM模型已保存: lgb_model.txt")

xgb_model.save_model('xgb_model.json')
print("✅ XGBoost模型已保存: xgb_model.json")

with open('feature_cols.pkl', 'wb') as f:
    pickle.dump(feature_cols, f)
print("✅ 特征列名已保存: feature_cols.pkl")

with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
print("✅ Label Encoders已保存: label_encoders.pkl")

# ============================================================
# Step 10: 结果汇总
# ============================================================
print("\n" + "=" * 70)
print("📊 模型训练结果汇总")
print("=" * 70)

print(f"""
┌────────────────────┬──────────────┐
│ 模型               │ RMSLE (验证集)│
├────────────────────┼──────────────┤
│ LightGBM           │ {lgb_rmsle:.5f}      │
│ XGBoost            │ {xgb_rmsle:.5f}      │
│ 模型融合 (权重{best_weight:.1f}/{1-best_weight:.1f})│ {best_ensemble_rmsle:.5f}      │
└────────────────────┴──────────────┘

特征数: {X.shape[1]}
训练数据量: {len(X_train):,} 行
验证数据量: {len(X_val):,} 行
""")

print("=" * 70)
print("✅ 第四阶段：模型训练完成！")
print("=" * 70)