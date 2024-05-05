import time
import pickle
import joblib
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn import preprocessing
import matplotlib as mpl
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
from sklearn.kernel_ridge import KernelRidge
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasRegressor



def main():
    start_time = time.time()
    fts = [40]
    random_states = [1382]
    def parallel_function(kk, random_state):
        avg_r2_score = class_gen_random_state(kk, 'CatBoost', random_state)
        return avg_r2_score
    results = Parallel(n_jobs=-1)(
        delayed(parallel_function)(kk, random_state) for kk in fts for random_state in random_states
    )
#    print("Results:", results)
def train_model(feature_train_scaled, y_train, model_name):
    if model_name == 'GBR':
        param_grid={'learning_rate': 0.01, 'max_depth': 8, 'max_features': 'log2', 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 2400}
        model = GradientBoostingRegressor(**param_grid)
    elif model_name == 'RFR':
        param_grid= {'bootstrap': True, 'max_depth': 50, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 500}
        model = RandomForestRegressor(**param_grid)
    elif model_name == 'KRR':
        model = KernelRidge(alpha=0.5, kernel='rbf')
    elif model_name == 'SVR':
        param_grid={'C': 1, 'epsilon': 0.01, 'gamma': 'scale', 'kernel': 'rbf'}
        model = SVR(**param_grid)
    elif model_name == 'LGBM':
        params={'learning_rate': 0.05, 'max_depth': 16, 'n_estimators': 400, 'num_leaves': 30}
        model = LGBMRegressor(**params)
    elif model_name == 'CatBoost':
        model = CatBoostRegressor(verbose=False)
    elif model_name == 'AdaBoost':
        #params={'learning_rate': 0.05, 'n_estimators': 400}
        #model = AdaBoostRegressor(**params)
        model=AdaBoostRegressor()
    elif model_name == 'XGB':
        #param_grid={'colsample_bylevel': 0.7, 'colsample_bytree': 0.6, 'eval_metric': 'mae', 'gamma': 0.0, 'learning_rate': 0.01, 'max_depth': 10, 'min_child_weight': 4, 'n_estimators': 400, 'objective': 'reg:gamma', 'reg_alpha': 0.01, 'reg_lambda': 0.01, 'subsample': 0.6000000000000001}
        params= {'colsample_bytree': 0.5, 'gamma': 0.0, 'learning_rate': 0.05, 'max_depth': 8, 'n_estimators': 800, 'reg_alpha': 0.01, 'reg_lambda': 0.1, 'subsample': 0.6000000000000001}
        params={'colsample_bytree': 0.9, 'gamma': 0.0, 'learning_rate': 0.05, 'max_depth': 4, 'n_estimators': 800, 'reg_alpha': 0.1, 'reg_lambda': 1, 'subsample': 0.7000000000000001}
        model = xgb.XGBRegressor(**params)
    elif model_name == 'NeuralNetwork':
        print("NN")

        def create_model():
            model = Sequential()
            model.add(Dense(32, activation='relu', input_shape=(feature_train_scaled.shape[1],)))
            model.add(Dense(32, activation='relu'))
            model.add(Dense(1))
            model.compile(loss='mean_squared_error', optimizer='adam')
            return model

        model = KerasRegressor(model=create_model, epochs=150, batch_size=64, verbose=0)

    model.fit(feature_train_scaled, y_train.values)
    return model
def class_gen_random_state(n, model_name, random_state):
    df = pd.read_csv('refrac_n_oqmd.csv')
    df = df.dropna()
    index_names = df[df['refractive'] >10.0].index
    df.drop(index_names, inplace=True)
  #  print(len(df))
    text_file = open("feature_refractive.txt", "r")
    #text_file = open("ft_temp_gb.txt", "r")
    feature_sorted = text_file.read()
    feature_sorted = feature_sorted.split("\n")
    features = feature_sorted[0:n]
    X = df[['mat_id'] + features]
    y = df['refractive']
    r2_scores = []
   # for random_state in random_states:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    x_scaler = MinMaxScaler(feature_range=(-1, 1))
    feature_train_scaled = x_scaler.fit_transform(X_train.values[:, 1:])
    model = train_model(feature_train_scaled, y_train, model_name)
    test_scaled=x_scaler.transform(X_test.values[:, 1:])
    y_pred = model.predict(test_scaled)
    r2 = r2_score(y_test, y_pred)
    mae=mean_absolute_error(y_test, y_pred)
    rmse=mean_squared_error(y_test, y_pred,squared = False)
    print("R2 MAE RMSE","   ",r2,mae,rmse)
   # file=open("gbr_pred.txt","w")
  #  plot_r2(y_test, y_pred,r2)
    # for k in zip(y_pred,y_test):
    #     file.write(f"{k[0]:<10}  {k[1]:>10}\n")
    # file.close()
    r2_scores.append(r2)
    return r2_scores
# def plot_r2(y_test, y_pred,r2):
#     mpl.rcParams['font.weight'] = 'bold'
#     mpl.rcParams['font.size'] = 20
#     fig,ax = plt.subplots()
#     x=np.linspace(0,8.0,1000)
#     plt.scatter(y_pred,y_test)
#     plt.plot(x,x,color='black',lw=3)
#     plt.xlabel(" Predicted\nrefractive Index",fontweight='bold',fontsize=20)
#     plt.ylabel("Ground Truth",fontweight='bold',fontsize=20)
#     plt.title("LGBM model",fontweight='bold',fontsize=20)
#     plt.xlim(0,8.0)
#     plt.ylim(0,8.0)
#     plt.text(0.8,7.0,f"R$^2$={round(r2,2)}")
#     for axis in ['top','bottom','left','right']:
#         ax.spines[axis].set_linewidth(3)
#     #plt.savefig("svr.png")
#     plt.savefig("lgbm.pdf", format="pdf", bbox_inches="tight")
if __name__ == '__main__':
    main()