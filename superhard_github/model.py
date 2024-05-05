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
    # random_states = [k for k in range(2,1500,30)]
    random_states=[872]
    def parallel_function(kk, random_state):
        avg_r2_score = class_gen_random_state(kk, 'CatBoost', random_state)
        return avg_r2_score
    results = Parallel(n_jobs=-1)(
        delayed(parallel_function)(kk, random_state) for kk in fts for random_state in random_states
    )
#    print("Results:", results)
def train_model(feature_train_scaled, y_train, model_name):

    if model_name == 'CatBoost':
        model = CatBoostRegressor(verbose=False)

    model.fit(feature_train_scaled, y_train.values)
    return model
def class_gen_random_state(n, model_name, random_state):
    df = pd.read_csv('superhard_oqmd.csv')
    df = df.dropna()
    index_names = df[df['hardness'] > 100.0].index
    df.drop(index_names, inplace=True)
  #  print(len(df))
    text_file = open("feature_rank1_oqmd.txt", "r")
    #text_file = open("ft_temp_gb.txt", "r")
    feature_sorted = text_file.read()
    feature_sorted = feature_sorted.split("\n")
    features = feature_sorted[0:n]
    X = df[['mat_id'] + features]
    y = df['hardness']
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
    print("feature count",n,"random_state:",random_state,"   ",r2,mae,rmse)
#    file=open("gbr_pred.txt","w")
#    plot_r2(y_test, y_pred,r2)
  #  plot_r2(y_test, y_pred,r2)
    # for k in zip(y_pred,y_test):
    #     file.write(f"{k[0]:<10}  {k[1]:>10}\n")
    # file.close()
    # r2_scores.append(r2)
    return r2_scores
def plot_r2(y_test, y_pred,r2):
    mpl.rcParams['font.weight'] = 'bold'
    mpl.rcParams['font.size'] = 20
    fig,ax = plt.subplots()
    x=np.linspace(0,90.0,1000)
    plt.scatter(y_pred,y_test)
    plt.plot(x,x,color='black',lw=3)
    plt.xlabel(" Predicted\n Hardness value",fontweight='bold',fontsize=20)
    plt.ylabel("Ground Truth",fontweight='bold',fontsize=20)
   # plt.title("LGBM model",fontweight='bold',fontsize=20)
    plt.xlim(0,90.0)
    plt.ylim(0,90.0)
    plt.text(7.0,67.0,f"R$^2$={round(r2,3)}")
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3)
    #plt.savefig("svr.png")
    plt.savefig("CatBoost.pdf", format="pdf", bbox_inches="tight")
if __name__ == '__main__':
    main()