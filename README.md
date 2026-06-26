# 🚕 纽约出租车行程时长预测
> 基于1786万条纽约出租车数据的行程时长预测项目
## 📌 项目简介
本项目基于纽约市TLC公开的出租车行程数据，构建机器学习模型预测单次行程的时长。
数据规模：1786万条行程记录
评估指标：RMSLE（Root Mean Squared Logarithmic Error）
最优成绩：0.052

## 🛠️ 技术栈

| 类别 | 技术 |
| 数据处理 | Python, Pandas, NumPy |
| 可视化 | Matplotlib, Seaborn |
| 机器学习 | Scikit-learn, LightGBM, XGBoost |

## 📁 项目结构
nyc-taxi-prediction/
├── New York Taxi.py # 主代码（清洗+特征工程+建模）
├── NYT model.py # 模型训练脚本
├── feature_importance.png # 特征重要性图
├── eda_duration_distribution.png # EDA分析图
├── data_cleaning_comparison.png # 清洗前后对比图
└── feature_engineering_distributions.png # 特征工程分布图

text

## 📊 数据处理流程
原始数据（1786万条）
↓
数据清洗（剔除异常值）
↓
特征工程（时间+速度特征）
↓
模型训练（LightGBM + XGBoost）
↓
模型融合（加权平均）
↓
RMSLE = 0.052 ✅

text

## 📈 模型效果

| 模型 | RMSLE |
| LightGBM | 0.0561 |
| XGBoost | 0.0533 |
| 模型融合 | 0.0520 |

## 📚 数据来源

[NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

数据时段：2024年1月 - 2024年6月

## 👨‍💻 作者
suze
🔗 GitHub: [suzesuze](https://github.com/suzesuze)
