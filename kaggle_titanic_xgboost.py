#1. データの取得と欠損値の確認
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import xgboost as xgb

le = LabelEncoder()

train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')
#trainデータとtestデータを１つにまとめる
data = pd.concat([train,test]).reset_index(drop=True)
#欠損値が含まれる行数を確認
train.isnull().sum()
test.isnull().sum()


#2. 欠損値の補完と特徴量の作成

#2.1Fareの補完と階級分け
#欠損値の補完
data['Fare'] = data['Fare'].fillna(data.query('Pclass==3 & Embarked=="S"')['Fare'].median())
#階級分けしたものを'Fare_bin'に入れる
data['Fare_bin'] = 0
data.loc[(data['Fare']>=10) & (data['Fare']<50), 'Fare_bin'] = 1
data.loc[(data['Fare']>=50) & (data['Fare']<100), 'Fare_bin'] = 2
data.loc[(data['Fare']>=100), 'Fare_bin'] = 3

#2.2 グループごとの生死の違い'Family_survival'の作成
#名前の名字を取得して'Last_name'に入れる
data['Last_name'] = data['Name'].apply(lambda x: x.split(",")[0])

data['Family_survival'] = 0.5 #デフォルトの値
#Last_nameとFareでグルーピング
for grp, grp_df in data.groupby(['Last_name', 'Fare']):

    if (len(grp_df) != 1):
        #(名字が同じ)かつ(Fareが同じ)人が2人以上いる場合
        for index, row in grp_df.iterrows():
            smax = grp_df.drop(index)['Survived'].max()
            smin = grp_df.drop(index)['Survived'].min()
            passID = row['PassengerId']

            if (smax == 1.0):
                data.loc[data['PassengerId'] == passID, 'Family_survival'] = 1
            elif (smin == 0.0):
                data.loc[data['PassengerId'] == passID, 'Family_survival'] = 0
            #グループ内の自身以外のメンバーについて
            #1人でも生存している → 1
            #生存者がいない(NaNも含む) → 0
            #全員NaN → 0.5

#チケット番号でグルーピング
for grp, grp_df in data.groupby('Ticket'):
    if (len(grp_df) != 1):
        #チケット番号が同じ人が2人以上いる場合
        #グループ内で1人でも生存者がいれば'Family_survival'を1にする
        for ind, row in grp_df.iterrows():
            if (row['Family_survival'] == 0) | (row['Family_survival']== 0.5):
                smax = grp_df.drop(ind)['Survived'].max()
                smin = grp_df.drop(ind)['Survived'].min()
                passID = row['PassengerId']
                if (smax == 1.0):
                    data.loc[data['PassengerId'] == passID, 'Family_survival'] = 1
                elif (smin == 0.0):
                    data.loc[data['PassengerId'] == passID, 'Family_survival'] = 0

#2.3 家族の人数を表す特徴量 'Family_size' の作成と階級分け
#Family_sizeの作成
data['Family_size'] = data['SibSp']+data['Parch']+1
#1, 2~4, 5~の3つに分ける
data['Family_size_bin'] = 0
data.loc[(data['Family_size']>=2) & (data['Family_size']<=4),'Family_size_bin'] = 1
data.loc[(data['Family_size']>=5) & (data['Family_size']<=7),'Family_size_bin'] = 2
data.loc[(data['Family_size']>=8),'Family_size_bin'] = 3

#2.4 名前の敬称 'Title' の作成
#名前の敬称を取得して'Title'に入れる
data['Title'] = data['Name'].map(lambda x: x.split(', ')[1].split('. ')[0])
#数の少ない敬称を統合
data['Title'].replace(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer', inplace=True)
data['Title'].replace(['Don', 'Sir',  'the Countess', 'Lady', 'Dona'], 'Royalty', inplace=True)
data['Title'].replace(['Mme', 'Ms'], 'Mrs', inplace=True)
data['Title'].replace(['Mlle'], 'Miss', inplace=True)
data['Title'].replace(['Jonkheer'], 'Master', inplace=True)

#2.5 チケット番号のラベリング
#数字のみのチケットと数字とアルファベットを含むチケットに分ける
#数字のみのチケットを取得
num_ticket = data[data['Ticket'].str.match('[0-9]+')].copy()
num_ticket_index = num_ticket.index.values.tolist()
#元のdataから数字のみのチケットの行を落とした残りがアルファベットを含むチケット
num_alpha_ticket = data.drop(num_ticket_index).copy()

#数字のみのチケットの階級分け
#チケット番号は文字列になっているので数値に変換
num_ticket['Ticket'] = num_ticket['Ticket'].apply(lambda x:int(x))

num_ticket['Ticket_bin'] = 0
num_ticket.loc[(num_ticket['Ticket']>=100000) & (num_ticket['Ticket']<200000),
               'Ticket_bin'] = 1
num_ticket.loc[(num_ticket['Ticket']>=200000) & (num_ticket['Ticket']<300000),
               'Ticket_bin'] = 2
num_ticket.loc[(num_ticket['Ticket']>=300000),'Ticket_bin'] = 3

#数字とアルファベットを含むチケットの階級分け
num_alpha_ticket['Ticket_bin'] = 4
num_alpha_ticket.loc[num_alpha_ticket['Ticket'].str.match('A.+'),'Ticket_bin'] = 5
num_alpha_ticket.loc[num_alpha_ticket['Ticket'].str.match('C.+'),'Ticket_bin'] = 6
num_alpha_ticket.loc[num_alpha_ticket['Ticket'].str.match('C\.*A\.*.+'),'Ticket_bin'] = 7
num_alpha_ticket.loc[num_alpha_ticket['Ticket'].str.match('F\.C.+'),'Ticket_bin'] = 8
num_alpha_ticket.loc[num_alpha_ticket['Ticket'].str.match('PC.+'),'Ticket_bin'] = 9
num_alpha_ticket.loc[num_alpha_ticket['Ticket'].str.match('S\.+.+'),'Ticket_bin'] = 10
num_alpha_ticket.loc[num_alpha_ticket['Ticket'].str.match('SC.+'),'Ticket_bin'] = 11
num_alpha_ticket.loc[num_alpha_ticket['Ticket'].str.match('SOTON.+'),'Ticket_bin'] = 12
num_alpha_ticket.loc[num_alpha_ticket['Ticket'].str.match('STON.+'),'Ticket_bin'] = 13
num_alpha_ticket.loc[num_alpha_ticket['Ticket'].str.match('W\.*/C.+'),'Ticket_bin'] = 14

data = pd.concat([num_ticket,num_alpha_ticket]).sort_values('PassengerId')

#2.6 Ageの補完と階級分け
#文字列になっている特徴量をラベリング
data['Sex'] = le.fit_transform(data['Sex']) #生存率予測に使うのでついでにラベリング
data['Title'] = le.fit_transform(data['Title'])
#Ageの予測に使う特徴量を'age_data'にいれる
age_data = data[['Age','Pclass','Family_size',
                 'Fare_bin','Title']].copy()
#Ageが欠損している行と欠損していない行に分ける
known_age = age_data[age_data['Age'].notnull()].values
unknown_age = age_data[age_data['Age'].isnull()].values

x = known_age[:, 1:]
y = known_age[:, 0]
#ランダムフォレストで学習
rfr = RandomForestRegressor(random_state=0, n_estimators=100, n_jobs=-1)
rfr.fit(x, y)
#予測値を元のデータフレームに反映する
age_predict = rfr.predict(unknown_age[:, 1:])
data.loc[(data['Age'].isnull()), 'Age'] = np.round(age_predict,1)

#Ageの階級分け
data['Age_bin'] = 0
data.loc[(data['Age']>18) & (data['Age']<=60),'Age_bin'] = 1
data.loc[(data['Age']>60),'Age_bin'] = 2

#いらない特徴量を落とす
data = data.drop(['PassengerId','Name','Age','SibSp','Parch','Ticket',
                  'Fare','Cabin','Embarked','Last_name','Family_size'], axis=1)

#統合させていたデータをtrainデータとtestデータに分ける
model_train = data[:891]
model_test = data[891:]

X = model_train.drop('Survived', axis=1)
Y = pd.DataFrame(model_train['Survived'])
x_test = model_test.drop('Survived', axis=1)


#3. xgboostでの予測
#パラメータを設定
params = {'objective':'binary:logistic',
          'max_depth':5,
          'eta': 0.1,
          'min_child_weight':1.0,
          'gamma':0.0,
          'colsample_bytree':0.8,
          'subsample':0.8}

num_round = 1000

logloss = []
accuracy = []

kf = KFold(n_splits=4, shuffle=True, random_state=0)
for train_index, valid_index in kf.split(X):
    x_train, x_valid = X.iloc[train_index], X.iloc[valid_index]
    y_train, y_valid = Y.iloc[train_index], Y.iloc[valid_index]
    #データフレームをxgboostに適した形に変換
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dvalid = xgb.DMatrix(x_valid, label=y_valid)
    dtest = xgb.DMatrix(x_test)
    #xgboostで学習
    model = xgb.train(params, dtrain, num_round,evals=[(dtrain,'train'),(dvalid,'eval')],
                      early_stopping_rounds=50)

    valid_pred_proba = model.predict(dvalid)
    #loglossを求める
    score = log_loss(y_valid, valid_pred_proba)
    logloss.append(score)
    #accuracyを求める
    #valid_pred_probaは確率値なので0と1に変換
    valid_pred = np.where(valid_pred_proba >0.5,1,0)
    acc = accuracy_score(y_valid, valid_pred)
    accuracy.append(acc)

print(f'log_loss:{np.mean(logloss)}')
print(f'accuracy:{np.mean(accuracy)}')

#predictで予測
y_pred_proba = model.predict(dtest)
y_pred= np.where(y_pred_proba > 0.5,1,0)
#データフレームを作成
submission = pd.DataFrame({'PassengerId':test['PassengerId'], 'Survived':y_pred})
submission.to_csv('titanic_xgboost.csv', index=False)
