#======== IMPORT =============
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import matplotlib.pyplot as plt


#========= READ DATASET =======
brain_tumor_data = pd.read_csv(r"Data/bt_dataset_t3_fixed.csv")
del brain_tumor_data["Unnamed: 0"]

#========= SVM CLASSIFIER ===========
X = brain_tumor_data.drop("Target", axis=1).values
Y = brain_tumor_data["Target"].values
ss = StandardScaler()
X = ss.fit_transform(X)

#========= SVM WITH PCA 2D ==================
pca = PCA(n_components=2)

principal_components = pca.fit_transform(X)
principalDF = pd.DataFrame(data=principal_components,
             columns=['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDF, brain_tumor_data[['Target']]], axis=1)

X = finalDf.drop("Target", axis=1)
Y = finalDf["Target"].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

svc = SVC(kernel="linear", probability=True)
svc.fit(X_train, Y_train)
score_linear2d = svc.score(X_test, Y_test)
print("=========== PCA 2D ===========")
print("ACCURACY WITH LINEAR KERNEL: " + str(score_linear2d))

svc = SVC(kernel="rbf", probability=True)
svc.fit(X_train, Y_train)
score_rbf2d = svc.score(X_test, Y_test)
print("ACCURACY WITH GAUSSIAN KERNEL: " + str(score_rbf2d))

svc = SVC(kernel="sigmoid", probability=True)
svc.fit(X_train, Y_train)
score_sigmoid2d = svc.score(X_test, Y_test)
print("ACCURACY WITH SIGMOID KERNEL: " + str(score_sigmoid2d))

#=========== SVM WITH PCA 3D ======================
