%pip install pandas
%pip install numpy
%pip install pip
%pip install matplotlib
%pip install seaborn
%pip install scipy
%pip install openpyxl
%pip list
%pip uninstall sequential
%pip install scikit-learn
%pip install xgboost
  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import os

#ML Model
from sklearn.model_selection import RandomizedSearchCV, train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge, ElasticNet, LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.base import clone

sensor_coords = {
    "R1": (1, 4), "R2": (2, 4), "R3": (3, 4),
    "R4": (4, 3), "R5": (4, 2), "R6": (4, 1),
    "R7": (1, 0), "R8": (2, 0), "R9": (3, 0),
    "R10": (0, 1), "R11": (0, 2), "R12": (0, 3)
}

force_coords={1:(3,1),2:(2,1),3:(1,1),4:(1,2),5:(2,2),6:(3,2),7:(3,3),8:(2,3),9:(1,1)}

folder_path='E:/KU/PhrModel/Contact probe of 8 mm diameter (9 position)-20250722T235649Z-1-001/Contact probe of 8 mm diameter (9 position)'

#input sampling
sample = 2
if sample==1:
    resistfile_loop1="E:/KU/PhrModel/Contact probe of 8 mm diameter (9 position)-20250722T235649Z-1-001/Contact probe of 8 mm diameter (9 position)/Sample_1/loop1"
    resistfile_loop2="E:/KU/PhrModel/Contact probe of 8 mm diameter (9 position)-20250722T235649Z-1-001/Contact probe of 8 mm diameter (9 position)/Sample_1/loop2"
    resistfile_loop3="E:/KU/PhrModel/Contact probe of 8 mm diameter (9 position)-20250722T235649Z-1-001/Contact probe of 8 mm diameter (9 position)/Sample_1/loop3"
    #raw/drive-download/02-07-2025/Contact probe of 8 mm diameter (9 position)/Force Sample_1/loop1/9 phr 6mm force_loop1_P1.csv
    fileforce_loop1="E:/KU/PhrModel/Contact probe of 8 mm diameter (9 position)-20250722T235649Z-1-001/Contact probe of 8 mm diameter (9 position)/Force Sample_1/loop1"
    fileforce_loop2="E:/KU/PhrModel/Contact probe of 8 mm diameter (9 position)-20250722T235649Z-1-001/Contact probe of 8 mm diameter (9 position)/Force Sample_1/loop2"
    fileforce_loop3="E:/KU/PhrModel/Contact probe of 8 mm diameter (9 position)-20250722T235649Z-1-001/Contact probe of 8 mm diameter (9 position)/Force Sample_1/loop3"
elif sample==2:
    resistfile_loop1="E:/KU/PhrModel/Contact probe of 8 mm diameter (9 position)-20250722T235649Z-1-001/Contact probe of 8 mm diameter (9 position)/Sample_2/loop1"
    resistfile_loop2="E:/KU/PhrModel/Contact probe of 8 mm diameter (9 position)-20250722T235649Z-1-001/Contact probe of 8 mm diameter (9 position)/Sample_2/loop2"
    resistfile_loop3="E:/KU/PhrModel/Contact probe of 8 mm diameter (9 position)-20250722T235649Z-1-001/Contact probe of 8 mm diameter (9 position)/Sample_2/loop3"
    resistfile_loop4="E:/KU/PhrModel/Contact probe of 8 mm diameter (9 position)-20250722T235649Z-1-001/Contact probe of 8 mm diameter (9 position)/Sample_2/loop4"
    resistfile_loop5="E:/KU/PhrModel/Contact probe of 8 mm diameter (9 position)-20250722T235649Z-1-001/Contact probe of 8 mm diameter (9 position)/Sample_2/loop5"
    #raw/drive-download/02-07-2025/Contact probe of 8 mm diameter (9 position)/Force Sample_1/loop1/9 phr 6mm force_loop1_P1.csv
    fileforce_loop1="E:/KU/PhrModel/Contact probe of 8 mm diameter (9 position)-20250722T235649Z-1-001/Contact probe of 8 mm diameter (9 position)/Force Sample_2/loop1"
    fileforce_loop2="E:/KU/PhrModel/Contact probe of 8 mm diameter (9 position)-20250722T235649Z-1-001/Contact probe of 8 mm diameter (9 position)/Force Sample_2/loop2"
    fileforce_loop3="E:/KU/PhrModel/Contact probe of 8 mm diameter (9 position)-20250722T235649Z-1-001/Contact probe of 8 mm diameter (9 position)/Force Sample_2/loop3"
    fileforce_loop4="E:/KU/PhrModel/Contact probe of 8 mm diameter (9 position)-20250722T235649Z-1-001/Contact probe of 8 mm diameter (9 position)/Force Sample_2/loop4"
    fileforce_loop5="E:/KU/PhrModel/Contact probe of 8 mm diameter (9 position)-20250722T235649Z-1-001/Contact probe of 8 mm diameter (9 position)/Force Sample_2/loop5"

#concent=9
#depth=6
#nloop=1
#pp=2
#force_file='{}/{} phr {}mm force_loop{}_P{}.csv'.format(force_folder_loop1,concent,depth,nloop,pp)
#force_dat=pd.read_csv(force_file)
def get_df_R1_12_w(folder_path,thickness,fileforce_folder,resistance_folder,outputR,num_loop):
    listforce=[]
    listpos=[]
    listdepth=[]
    listpos=[]
    listr1=[]
    listr2=[]
    listr3=[]
    listr4=[]
    listr5=[]
    listr6=[]
    listr7=[]
    listr8=[]
    listr9=[]
    listr10=[]
    listr11=[]
    listr12=[]
    listfiles=os.listdir(folder_path)
    #print(os.listdir("{}".format(resistance_folder))[0].split()[0])
    
    concentrate=int(os.listdir("{}".format(resistance_folder))[0].split()[0])
    # print(" type num_loop: \t")
    # print(type(num_loop))
    
    #fileforce_name="loop 1.csv"
    #loadforce=pd.read_csv("{}/{}".format(folder_path,fileforce_name))
    #raw/drive-download/13-06-2025/Contact probe of 8 mm diameter (9 position)/Force/loop1/9 phr 6mm force_loop1_P1.csv
    
    for pp in range(1,10):
        #raw/drive-download/02-07-2025/Contact probe of 8 mm diameter (9 position)/Force Sample_1/loop1/9 phr 6mm force_loop1_P1.csv
        fileforce_name="{}/{} phr {}mm force_loop{}_P{}.csv".format(fileforce_folder,concentrate,thickness,num_loop,pp)
        loadforce=pd.read_csv(fileforce_name)
        
    #print(loadforce)
        for i in range(1,11):
            listdepth.append(5*i*thickness/100)
            #concent="{} {} {}mm {}_{}_".format(listfiles[0].split()[0],listfiles[0].split()[1],thickness,listfiles[0].split()[-1][:-3],5*i)
            #fileresist="{}/{}/dat00001.csv".format(folder_path,concent)
            #raw/drive-download/13-06-2025/Contact probe of 3 cm diameter (1 position)/loop1/9 phr 6mm loop1_R1-12_5
            #raw/drive-download/13-06-2025/Contact probe of 8 mm diameter (9 position)/loop1/9 phr 6mm loop1_R1-12_P1_5/dat00001.csv
            fileresist="{}/{} phr {}mm loop{}_{}_P{}_{}/dat00001.csv".format(resistance_folder,concentrate,thickness,num_loop,outputR,pp,5*i)
            
            # print("fileresist: ", type(fileresist))

            getresist=pd.read_csv(fileresist)
            # print("getresist: \n")
            # print(getresist.iloc[:,2:].values)

            
            
            getmeans=getresist.iloc[:,2:].mean().values
            # print(f"point {pp}: means {getmeans}")
            # if (pp==4): 
            #    plt.plot(getresist.iloc[:,2:])
            #    plt.plot(np.ones(len(getresist.iloc[:,2:]))*getmeans)
            #    plt.show()
            
            listr1.append(getmeans[0])
            listr2.append(getmeans[1])
            listr3.append(getmeans[2])
            listr4.append(getmeans[3])
            listr5.append(getmeans[4])
            listr6.append(getmeans[5])
            listr7.append(getmeans[6])
            listr8.append(getmeans[7])
            listr9.append(getmeans[8])
            listr10.append(getmeans[9])
            listr11.append(getmeans[10])
            listr12.append(getmeans[11])
            listpos.append(pp)
            test_path=loadforce.iloc[2:,3*i-2].dropna().astype(float)
            listforce.append(max(test_path))
    df=pd.DataFrame({"depth":listdepth,"force":listforce,"position":listpos,"r1":listr1,"r2":listr2,"r3":listr3,"r4":listr4,"r5":listr5,"r6":listr6,"r7":listr7,"r8":listr8,"r9":listr9,"r10":listr10,"r11":listr11,"r12":listr12})
    
    #     # for col in range(3):
    #     #     print(f"col: {col}")
    #     fig, ax = plt.subplots()
    #     # ax = axes[1, int(num_loop)]
    #     ax.boxplot(data = df_normalized)
    #     ax.set_title('Distribution of Normalized Features (Min-Max Scaling)')    
    #     ax.xlabel('Record Index (loop: {})'.format(num_loop))
    #     ax.ylabel('Resistance (Ohm)')

    #     # The BENEFIT: The legend is created automatically and correctly.
    return df

def plot_Pos(df, num_loop):
    x = 0
    # print(df)
    row = 0
    col = 0
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(46, 36), sharey=False)

    for Pos in range(1, 10):
        Pos = int(Pos * 10)
        # print(f"row: {row}")
        # print(f"col: {col}")
        # print(f"x: {x}")
        # print(f"pos: {Pos}")
        df_pos = pd.DataFrame(df.iloc[x:Pos,3:])
        # print(df_pos)
        axs[row, col].boxplot(df_pos)
        axs[row, col].set_xlabel(f"Index R /Position: {Pos/10}", fontsize=16)
        axs[row, col].set_ylabel("Ohm", fontsize=16)
        Pos = int(Pos / 10)
        row = int(Pos // 3)
        col = int(Pos % 3)
        x = x + 10
    for ax in axs.flat:
        ax.grid(True, linestyle='-', which='major', color='lightgrey',
               alpha=0.5)   
    plt.suptitle(f"sample {sample}/loop {num_loop}/Distribution of Normalized Features (Min-Max Scaling) **Relation Of Position and index R**", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show

def norm_nPos(df, num_loop):
    x = 0
    y = 0
    # print(f"loop: {num_loop}")
    # print(df)
    dist_list = []
    for Pos in range(1, 10):
        # print(f"position: {Pos}")
        Pos = int(Pos * 10)
        # print(f"x: {x}")
        # print(f"pos: {Pos}")
        df_pos = pd.DataFrame(df.iloc[x:Pos,3:])
        for r in range(0, 12):
            # print(f"R{r+1}")
            df_iR = df_pos.iloc[:, r] #for index R and position
            # print(df_iR)
            Q1n = np.quantile(df_iR, 0.25)
            Q3n = np.quantile(df_iR, 0.75)
            delta_Q = Q3n - Q1n
            min = Q1n - 1.5 * delta_Q
            max = Q3n + 1.5 * delta_Q
            # print(f"Q1: {Q1n}, \tQ3: {Q3n}, \tmin: {min}, \tmax: {max}\n")
            # print(df_pos.iloc[r, r])
            # if df_pos.iloc[r, r] >
            summary_data = {
                'nPos' : int(Pos/10),
                'index_r' : r+1,
                'Q1' : Q1n,
                'Q3' : Q3n,
                'min' : min,
                'max' : max
                }
            dist_list.append(summary_data)
            # print(summary_data)
        x = x + 10
        
        # print(f"Q3: {Q3n}")
        # print(f"min: {min}")
        # print(f"max: {max} \n")
        # show_table = pd.DataFrame()
        # print(pd.DataFrame({'Q1' : Q1n, 'Q3' : Q3n, 'min' : min, 'max' : max}))
        # Pos = int(Pos / 10)
    
    # plot_Pos(df, num_loop)
    # print(pd.DataFrame(dist_list))
    # print()
    return pd.DataFrame(dist_list)


def cleandf(df, num_loop, norm):
    pd.set_option('display.width', None)
    df_cp = df.copy()
    # df_cp.to_excel("E:/KU/PhrModel/raw.xlsx")
    # df_norm = norm.transpose()
    # print(df_cp)
    # print(norm)
    # (pd.DataFrame(norm)).to_excel("E:/KU/PhrModel/norm.xlsx")
    # outlier_list = []
    drop_row = []
    for aPos in range(0, 90):
        for index, rule in norm.iterrows():
            # print(index)
            # print(rule)
            nPos = int(rule["nPos"])
            # print(f"nPos:{nPos}")
            r = int(rule["index_r"])
            # print(f"r{r}")
            Q1 = rule["Q1"]
            Q3 = rule["Q3"]
            min = rule["min"]
            max = rule["max"]
            if df_cp.loc[int(aPos), "position"] == nPos:
                # print(f"nPos:{nPos}")
                # print(f"r{r}")
                # print(f"depth:{df_cp.loc[aPos, "depth"]}")
                # if(r == 9):
                # print(f"aPos:{aPos}\tposition:{df_cp.loc[int(aPos), "position"]}\tnPos:{nPos}\tr{r}\t{df_cp.loc[aPos, f"r{r}"]} ?? min:{min} or max:{max}")
                if df_cp.loc[aPos, f"r{r}"] > max or df_cp.loc[aPos, f"r{r}"] < min:
                    drop_row.append(aPos)
                    # break
                    # print(f"position:{df_cp.loc[int(aPos), "position"]}\tnPos:{nPos}\tr{r}")
                    # print(df_cp.loc[aPos, f"r{r}"])
                    # print()
                    # summary_data = {
                    #     'position' : f"{df_cp.loc[int(aPos), "position"]}",
                    #     'index_r' : f"r{r}",
                    #     'r_value' : f"{df_cp.loc[aPos, f"r{r}"]}"
                    #     }
                    # outlier_list.append(summary_data)
    print(set(drop_row)) # grouping the same number
    print(f"amount of outlier = {len(set(drop_row))}") # grouping the same number
    df_cp.drop(set(drop_row), inplace=True)
    # pd.DataFrame(outlier_list).to_excel("outlier.xlsx")
    # df_cp.to_excel("E:/KU/PhrModel/clean.xlsx")
    # print(df_cp)
    # print("Done")
    # print()
    return df_cp


if sample==1:
    df_loop1=get_df_R1_12_w(folder_path,6,fileforce_loop1,resistfile_loop1,"R1-12",1)
    df_loop2=get_df_R1_12_w(folder_path,6,fileforce_loop2,resistfile_loop2,"R1-12",2)
    df_loop3=get_df_R1_12_w(folder_path,6,fileforce_loop3,resistfile_loop3,"R1-12",3)
    norm_Pos_loop1 = norm_nPos(df_loop1, num_loop=1)
    norm_Pos_loop2 = norm_nPos(df_loop2, num_loop=2)
    norm_Pos_loop3 = norm_nPos(df_loop3, num_loop=3)
    clean_loop1 = cleandf(df_loop1, num_loop=1, norm=norm_Pos_loop1)
    clean_loop2 = cleandf(df_loop2, num_loop=2, norm=norm_Pos_loop2)
    clean_loop3 = cleandf(df_loop3, num_loop=3, norm=norm_Pos_loop3)
    
elif sample==2:
    df_loop1=get_df_R1_12_w(folder_path,6,fileforce_loop1,resistfile_loop1,"R1-12",1)
    df_loop2=get_df_R1_12_w(folder_path,6,fileforce_loop2,resistfile_loop2,"R1-12",2)
    df_loop3=get_df_R1_12_w(folder_path,6,fileforce_loop3,resistfile_loop3,"R1-12",3)
    df_loop4=get_df_R1_12_w(folder_path,6,fileforce_loop4,resistfile_loop4,"R1-12",4)
    df_loop5=get_df_R1_12_w(folder_path,6,fileforce_loop5,resistfile_loop5,"R1-12",5)
    norm_Pos_loop1 = norm_nPos(df_loop1, num_loop=1)
    norm_Pos_loop2 = norm_nPos(df_loop2, num_loop=2)
    norm_Pos_loop3 = norm_nPos(df_loop3, num_loop=3)
    norm_Pos_loop4 = norm_nPos(df_loop4, num_loop=4)
    norm_Pos_loop5 = norm_nPos(df_loop5, num_loop=5)
    clean_loop1 = cleandf(df_loop1, num_loop=1, norm=norm_Pos_loop1)
    clean_loop2 = cleandf(df_loop2, num_loop=2, norm=norm_Pos_loop2)
    clean_loop3 = cleandf(df_loop3, num_loop=3, norm=norm_Pos_loop3)
    clean_loop4 = cleandf(df_loop4, num_loop=4, norm=norm_Pos_loop4)
    clean_loop5 = cleandf(df_loop5, num_loop=5, norm=norm_Pos_loop5)

# === Prepare training and test sets ===
df_train = pd.concat([clean_loop1, clean_loop2,clean_loop3,clean_loop4], ignore_index=True)
df_test = df_loop5

features = [f"r{i}" for i in range(1, 13)]
targets = ["depth", "force", "position"]

X_train = df_train[features]
y_train = df_train[targets]
X_test = df_test[features]
y_test = df_test[targets]

# === Apply polynomial features for MLP only ===
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# === Train models ===

## MLP Regressor
mlp_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=3000, random_state=42))
])
mlp_model = MultiOutputRegressor(mlp_pipeline)
mlp_model.fit(X_train_poly, y_train)
y_pred_mlp = mlp_model.predict(X_test_poly)
r2_mlp = [r2_score(y_test.iloc[:, i], y_pred_mlp[:, i]) for i in range(3)]

## Random Forest
rf_model = MultiOutputRegressor(RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42))
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
r2_rf = [r2_score(y_test.iloc[:, i], y_pred_rf[:, i]) for i in range(3)]

## XGBoost (simplified for speed)
xgb_model = MultiOutputRegressor(XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, objective='reg:squarederror'))
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
r2_xgb = [r2_score(y_test.iloc[:, i], y_pred_xgb[:, i]) for i in range(3)]

## Logistic Regression for position classification
clf = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000, multi_class='multinomial'))
])
clf.fit(X_train, y_train['position'].astype(int))
pos_pred = clf.predict(X_test)
acc_pos = accuracy_score(y_test['position'].astype(int), pos_pred)

# === Plot actual vs predicted ===
fig, axes = plt.subplots(3, 3, figsize=(18, 12))
model_names = ["MLP", "Random Forest", "XGBoost"]
predictions = [y_pred_mlp, y_pred_rf, y_pred_xgb]
r2_scores = [r2_mlp, r2_rf, r2_xgb]

for col in range(3):  # model
    for row in range(3):  # target
        ax = axes[row, col]
        ax.scatter(y_test.iloc[:, row], predictions[col][:, row], alpha=0.7)
        ax.plot([y_test.iloc[:, row].min(), y_test.iloc[:, row].max()],
                [y_test.iloc[:, row].min(), y_test.iloc[:, row].max()], 'r--')
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{targets[row]} | {model_names[col]} (R²: {r2_scores[col][row]:.3f})")

plt.tight_layout()
plt.show()

# === Print R² scores and accuracy ===
print("\nR² Scores by Model and Target")
for i, target in enumerate(targets):
    print(f"Target: {target.upper()}")
    print(f"  MLP:           R² = {r2_mlp[i]:.4f}")
    print(f"  Random Forest: R² = {r2_rf[i]:.4f}")
    print(f"  XGBoost:       R² = {r2_xgb[i]:.4f}")
print(f"\nPosition Classification Accuracy (LogReg): {acc_pos:.4f}")
