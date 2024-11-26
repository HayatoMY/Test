import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# CSVデータの読み込み
df_cor = pd.read_csv("sample/csv/check_correlation_2.csv")

# Survivedとの相関を計算
correlation = df_cor.corr()['Survived'].drop('Survived')

# ヒートマップで相関を可視化（全体の相関）
plt.figure(figsize=(10, 8))
sns.heatmap(df_cor.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Heatmap of Feature Correlations")
plt.show()