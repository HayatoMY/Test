import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#CSVファイルを読み込む
df_train = pd.read_csv('csv/train.csv')

#読み込んだCSVを確認する
#print(df_train.head())

#平均値、標準偏差、最小、最大値の確認
#print(df_train.describe())

df_train = df_train.dropna(axis=1)

#欠損している項目の確認
#print(df_train.isnull().sum())

#生存者・死亡者数（ヒストグラム）
#plt.hist(df_train['Survived'], bins=3)

#チケットの価格分布（ヒストグラム）
#plt.hist(df_train['Fare'], bins=15)

#性別分布（ヒストグラム）
#plt.hist(df_train['Sex'], bins=3)

#グラフの出力
#plt.show()

# クロス集計+割合/パラメータ表示（normalize）
df_survived = pd.crosstab(df_train['Sex'], df_train['Survived'],normalize='columns')

# 生存率・死亡率の男女比を取得
# 生存者の男女比
df_survived_1 = df_survived[1].values

# 死亡者の男女比
#df_survived_0 = df_survived[0].values.tolist()

# 生存率の男女比のグラフ化
plt.bar(x=np.array(['female','male']), height=df_survived_1)
plt.show()
