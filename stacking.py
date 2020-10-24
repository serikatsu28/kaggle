import pandas as pd
import numpy as np
import os, random
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.linear_model import LogisticRegression

le = LabelEncoder()
sc = StandardScaler()


#乱数を固定する関数
def reset_seed(seed):
    os.environ['PYTHONHASHSEED'] = '0'
    random.seed(seed) # random関数のシードを固定
    np.random.seed(seed) # numpyのシードを固定
    tf.random.set_seed(seed) # tensorflowのシードを固定

#乱数を固定
reset_seed(28)

#データの取得
train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')
#trainデータとtestデータを統合
data = pd.concat([train,test]).reset_index(drop=True)

#Sexのラベリング
data['Sex'] = le.fit_transform(data['Sex'])

#Fareの処理
#欠損値を補完
data['Fare'] = data['Fare'].fillna(data.query('Pclass==3 & Embarked=="S"')['Fare'].median())
#階級分け
data['Fare_bin'] = 0 #デフォルト値
data.loc[(data['Fare']>=10) & (data['Fare']<50), 'Fare_bin'] = 1
data.loc[(data['Fare']>=50) & (data['Fare']<100), 'Fare_bin'] = 2
data.loc[(data['Fare']>=100), 'Fare_bin'] = 3
#標準化
data['Fare_std'] = sc.fit_transform(data[['Fare']])

#家族の人数'Family_size'の作成
data['Family_size'] = data['SibSp']+data['Parch']+1
data['Family_size_bin'] = 0 #デフォルト値
data.loc[(data['Family_size']>=2) & (data['Family_size']<=4),'Family_size_bin'] = 1
data.loc[(data['Family_size']>=5) & (data['Family_size']<=7),'Family_size_bin'] = 2
data.loc[(data['Family_size']>=8),'Family_size_bin'] = 3

#名前の敬称'Title'の作成
data['Title'] = data['Name'].map(lambda x: x.split(', ')[1].split('. ')[0])
data['Title'].replace(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer', inplace=True)
data['Title'].replace(['Don', 'Sir',  'the Countess', 'Lady', 'Dona'], 'Royalty', inplace=True)
data['Title'].replace(['Mme', 'Ms'], 'Mrs', inplace=True)
data['Title'].replace(['Mlle'], 'Miss', inplace=True)
data['Title'].replace(['Jonkheer'], 'Master', inplace=True)

#グループごとの生存の違い'Family_survival'の作成
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
            #自身以外のメンバーについて
            #1人でも生存している→1
            #生存者がいない(NaNも含む)→0
            #全員NaN→0.5

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

#ランダムフォレストを使ったAgeの欠損値補完と階級分け
#Ageの予測に使う特徴量を'age_data'にいれる
age_data = data[['Age','Pclass','Family_size','Fare_bin','Title']].copy()
#文字列になっている特徴量をラベリング
age_data['Title'] = le.fit_transform(age_data['Title'])
#Ageが欠損している行と欠損していない行に分ける
known_age = age_data[age_data['Age'].notnull()].values
unknown_age = age_data[age_data['Age'].isnull()].values

x = known_age[:, 1:]
y = known_age[:, 0]
#ランダムフォレストで学習
rfr = RandomForestRegressor(random_state=28, n_estimators=100, n_jobs=-1)
rfr.fit(x, y)
#予測値を元のデータフレームに反映する
age_predict = rfr.predict(unknown_age[:, 1:])
data.loc[(data['Age'].isnull()), 'Age'] = np.round(age_predict,1)

#Ageの階級分け
data['Age_bin'] = 0
data.loc[(data['Age']>18) & (data['Age']<=60),'Age_bin'] = 1
data.loc[(data['Age']>60),'Age_bin'] = 2

#チケットの種類ごとにラベリング
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

#不要な特徴量を落とす
#xgboostとランダムフォレスト用のデータフレーム
data_xr = data.drop(['PassengerId','Name','Age','SibSp','Parch','Ticket',
                     'Fare','Cabin','Embarked','Fare_std','Family_size','Last_name'], axis=1)
data_xr['Title'] = le.fit_transform(data_xr['Title'])

#ニューラルネット用のデータフレーム
data_nn = data.drop(['PassengerId','Name','Age','SibSp','Parch','Ticket','Fare','Cabin',
                     'Embarked','Fare_bin','Family_size','Last_name','Ticket_bin'], axis=1)
data_nn = pd.get_dummies(data=data_nn, columns=['Title','Pclass','Family_survival'])

#trainデータとtestデータに分ける
model_train_xr = data_xr[:891]
model_test_xr = data_xr[891:]
#特徴量と目的変数に分ける
X_xr = model_train_xr.drop('Survived', axis=1)
Y_xr = pd.DataFrame(model_train_xr['Survived'])
x_test_xr = model_test_xr.drop('Survived', axis=1)
#trainデータとvalidデータに分ける
x_train_xr, x_valid_xr, y_train_xr, y_valid_xr = train_test_split(X_xr, Y_xr, test_size=0.3, random_state=28)

#trainデータとtestデータに分ける
model_train_nn = data_nn[:891]
model_test_nn = data_nn[891:]
#特徴量と目的変数に分ける
X_nn = model_train_nn.drop('Survived', axis=1)
Y_nn = pd.DataFrame(model_train_nn['Survived'])
x_test_nn = model_test_nn.drop('Survived', axis=1)
#trainデータとvalidデータに分ける
x_train_nn, x_valid_nn, y_train_nn, y_valid_nn = train_test_split(X_nn, Y_nn, test_size=0.3, random_state=28)


#モデルの構築と予測
valid_pred_list = [] # x_validでの予測値を入れる
test_pred_list = [] # x_testでの予測値を入れる

#xgboostでの予測
#パラメータを設定
xgb_params = {'objective':'binary:logistic',
          'max_depth':5,
          'eta': 0.1,
          'min_child_weight':1.0,
          'gamma':0.0,
          'colsample_bytree':0.8,
          'subsample':0.8}
num_round=1000

#データフレームをxgboostに適した形に変換
dtrain = xgb.DMatrix(x_train_xr, label=y_train_xr)
dvalid = xgb.DMatrix(x_valid_xr, label=y_valid_xr)
dtest = xgb.DMatrix(x_test_xr)

#xgboostで学習
xgb_model = xgb.train(xgb_params, dtrain, num_round,
                      evals=[(dtrain,'train'),(dvalid,'eval')],early_stopping_rounds=50)

#validデータでの予測
valid_pred_proba = xgb_model.predict(dvalid)
xgb_valid_pred = np.where(valid_pred_proba >0.5,1,0)
valid_pred_list.append(xgb_valid_pred)

#testデータでの予測
test_pred_proba = xgb_model.predict(dtest)
xgb_test_pred = np.where(test_pred_proba >0.5,1,0)
test_pred_list.append(xgb_test_pred)

#ランダムフォレストでの予測
#データフレームをnｄarrayに変換
x_train_xr = np.array(x_train_xr)
x_valid_xr = np.array(x_valid_xr)
y_train_xr = np.array(y_train_xr).ravel()
y_valid_xr = np.array(y_valid_xr).ravel()
x_test_xr = np.array(x_test_xr)

#パラメータ設定
rfc_params = {
    "n_estimators" : [5, 10, 15, 20, 30, 50, 75, 100],
    "min_samples_split" : [2, 3, 5, 10, 15, 20, 30],
    "max_depth" : [3, 5, 10, 15, 20, 30],
    "criterion" : ["gini"],
    "random_state" : [28],
    "verbose" : [False]
}

#ランダムフォレストでGridSearch&学習
rfc_model = RandomForestClassifier()
gscv = GridSearchCV(rfc_model, rfc_params, cv=4)
gscv.fit(x_train_xr, y_train_xr)

#validデータでの予測
rfc_valid_pred = gscv.predict(x_valid_xr).astype(int)
valid_pred_list.append(rfc_valid_pred)

#testデータでの予測
rfc_test_pred = gscv.predict(x_test_xr).astype(int)
test_pred_list.append(rfc_test_pred)

#ニューラルネットでの予測
#モデルの初期化
model = Sequential()
#層の構築
model.add(Dense(12, activation='relu', input_dim=16))
model.add(Dropout(0.2))
model.add(Dense(8, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
#モデルの構築
model.compile(optimizer = 'adam', loss='binary_crossentropy', metrics='acc')
#モデルでの学習
log = model.fit(x_train_nn, y_train_nn, epochs=5000, batch_size=32,verbose=1,
                callbacks=[EarlyStopping(monitor='val_loss',min_delta=0,patience=100,verbose=1)],
                validation_split=0.3)

#validデータでの予測
nn_valid_pred = model.predict_classes(x_valid_nn).reshape(-1)
valid_pred_list.append(nn_valid_pred)
#testデータでの予測
nn_test_pred = model.predict_classes(x_test_nn).reshape(-1)
test_pred_list.append(nn_test_pred)


#ロジスティック回帰での予測
#リストからndarrayに変換
valid_preds = np.column_stack(valid_pred_list)
test_preds = np.column_stack(test_pred_list)
print(valid_preds.shape)
print(test_preds.shape)

#y_validを1次元配列に変換
y_valid = np.array(y_valid_nn).reshape(-1).astype(int)
#ロジスティック回帰での学習＆予測
meta_model = LogisticRegression(solver='lbfgs', max_iter=10000)
meta_model.fit(valid_preds, y_valid)
meta_pred = meta_model.predict(test_preds).astype(int)

#提出用データフレームの作成
submission = pd.DataFrame({'PassengerId':test['PassengerId'], 'Survived':meta_pred})
submission.to_csv('titanic_stacking.csv', index=False)
