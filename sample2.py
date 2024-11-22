import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#CSVファイルを読み込む
df_train = pd.read_csv('sample/csv/train.csv')
df_test = pd.read_csv('sample/csv/test.csv')

#性別を数値（0,1,2,3）に変更　⇨ map関数を使う　
'''map関数後でちゃんとインプットする'''

df_train['Sex'] = df_train['Sex'].map({'male':0 , 'female':1})
df_test['Sex'] = df_test['Sex'].map({'male':0 , 'female':1})

# 名前（Name）から敬称（Title）を抽出
df_train['Title'] = df_train['Name'].str.extract(' ([A-Za-z]+)\\.', expand=False)
df_test['Title'] = df_test['Name'].str.extract(' ([A-Za-z]+)\\.', expand=False)

# 敬称をカテゴリごとに数値に変換
title_map = {
    'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4,
    'Dr': 5, 'Rev': 6, 'Col': 7, 'Major': 7,
    'Mlle': 2, 'Countess': 3, 'Ms': 2, 'Lady': 3,
    'Jonkheer': 7, 'Don': 7, 'Sir': 7, 'Mme': 3,
    'Capt': 7, 'Dona': 3
}

df_train['Title'] = df_train['Title'].map(title_map).fillna(0)
df_test['Title'] = df_test['Title'].map(title_map).fillna(0)

x = df_train[['Sex','Pclass','SibSp','Parch','Title']]
y = df_train['Survived']

# トレーニングデータとテストデータに分割
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0)

# ランダムフォレスト回帰モデルの作成
model = RandomForestRegressor(n_estimators=100, random_state=None)
model.fit(x_train, y_train)

# テストデータでの予測
predictions = model.predict(x_test)

print(model.score(x_test,y_test))