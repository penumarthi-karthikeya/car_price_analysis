import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))
    ax1 = sns.kdeplot(RedFunction, color="r", label=RedName,fill=True)
    ax2 = sns.kdeplot(BlueFunction, color="b", label=BlueName, fill=True)
    plt.title(Title)
    plt.xlabel('Price (in ₹Ruppes(Lacs))')
    plt.ylabel('Proportion of Cars')
    plt.xlim(0,)
    plt.show()
    plt.close()
def plots(X,xlabelname,df,Y=None,ylabelname=None):
    Y=df["price"]
    ylabelname="price"
    sns.set_style("whitegrid")
    sns.boxplot(x=X,y=Y,data=df)
    plt.title("Seaborn Boxplot!")
    plt.grid("50x50")
    plt.figure(figsize =(10, 10))
    plt.title("Scatterplot!")
    plt.scatter(X,Y)
    plt.xlabel(xlabelname)
    plt.ylabel(ylabelname)
    plt.show()
def pearsoncorelation(x):
    X=df[x]
    Y=df["price"]
    p_e,p_v=stats.pearsonr(X,Y)
    t="Correlation between {} and price"
    print(t.format(x))
    print("Pearson_correlaton value:->",round(p_e,3))
    print("P-Value:------------------>",p_v)
    print("\n")
path="K:\\Python\\carprice.csv"
df=pd.read_csv(path)
df=pd.read_csv(path)
df=df.replace('?',0)
# print("Dataframe Before Pre-processing\n")
# print(df.head(25))
ss=MinMaxScaler()
numeric_col=["wheel-base","length","width","height","curb-weight","engine-size","bore","stroke","compression-ratio","horsepower","peak-rpm","city-mpg","highway-mpg"]
for x in numeric_col:
    df[x]=df[x].astype(float)
    m=df[x].mean()
    df[x].replace(0,m,inplace=True)
    df[x]=round(df[x],2)
    df[x]=ss.fit_transform(df[x].values.reshape(-1,1))
    df[x]=round(df[x],2)
df["price"]=df["price"].astype(float)
mea=df["price"].mean()
df["price"]=df["price"].replace(0,mea)
df["price"]=df["price"]*80
df["price"]=round(df["price"],2)
le=LabelEncoder()
object_col=["normalized-losses","make","fuel-type","aspiration","num-of-doors","body-style","drive-wheels","engine-location","fuel-system","engine-type","num-of-cylinders"]
for x in object_col:
    df[x]=df[x].astype(str)
    df[x]=le.fit_transform(df[x])
# print("Dataframe Afther Pre-processing\n")
# print(df.head(25))
x=df.drop(["price"],axis=1)
y=df["price"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
pr=PolynomialFeatures(degree=3)
x_train_poly=pr.fit_transform(x_train)
x_test_poly=pr.transform(x_test)
ridge = Ridge(alpha=100)
ridge.fit(x_train_poly, y_train)
y_pred = ridge.predict(x_test_poly)
y_pred_=ss.inverse_transform(y_pred.reshape(-1,1))
y_test_np = y_test.values
y_pred_np = y_pred_.flatten()
for i in numeric_col:
    pearsoncorelation(i)
    # plots(X=df[i],xlabelname=i,df=df)
for i in object_col:
    pearsoncorelation(i)
print("Mean_Of_Co-effcients:--->",round(np.mean(ridge.coef_),3))
print("Intercept_Value:-------->",round(ridge.intercept_,3))
print("\n")
print("R_Squared_Error:-------->",round(ridge.score(x_train_poly,y_train),3))#R_Squared
score=cross_val_score(ridge,x,y,cv=5)
print("Cross_Vadilation_Score:->",round(np.mean(score),3))
pscore=cross_val_predict(ridge,x,y,cv=5)
print("Cross_Prediction_Score:->",round(np.mean(pscore),3))
Title = 'Distribution  Plot of  Actual and Predicated Values'
DistributionPlot(y_test,y_pred, "Actual Values (Train)", "Predicted Values (Train)", Title)


