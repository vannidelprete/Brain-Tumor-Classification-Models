#======== IMPORT =============
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

#========= READ DATASET =======
brain_tumor_data = pd.read_csv(r"Data/bt_dataset_t3_fixed.csv")
del brain_tumor_data["Unnamed: 0"]
#========= PLOTTING HEATMAP ==========
sns.heatmap(brain_tumor_data.corr(), xticklabels=brain_tumor_data.columns, yticklabels=brain_tumor_data.columns)
plt.show()

#======== LOGISTIC REGRESSION WITH ALL FEATURE =========
X = brain_tumor_data.drop("Target", axis=1).values
Y = brain_tumor_data["Target"].values
ss = StandardScaler()
X = ss.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

log_reg = LogisticRegression(solver='lbfgs')
log_reg.fit(X_train, Y_train)
Y_pred = log_reg.predict(X_test)
Y_pred_prob = log_reg.predict_proba(X_test)

acc = accuracy_score(Y_test, Y_pred)
log_loss_score = log_loss(Y_test, Y_pred_prob)

print("ACCURACY: " + str(acc) + "\n")
print("LOG LOSS: " + str(log_loss_score) + "\n")

#======== LOGISTIC REGRESSION WITH PCA 2D =========

X = brain_tumor_data.drop('Target', axis=1).values
X = ss.transform(X)

pca = PCA(n_components=2)
principal_components = pca.fit_transform(X)
principalDF = pd.DataFrame(data=principal_components,
             columns=['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDF, brain_tumor_data[['Target']]], axis=1)

X = finalDf.drop("Target", axis=1)
Y = finalDf["Target"].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)


log_reg.fit(X_train, Y_train)
Y_pred = log_reg.predict(X_test)
Y_pred_prob = log_reg.predict_proba(X_test)

acc = accuracy_score(Y_test, Y_pred)
log_loss_score = log_loss(Y_test, Y_pred_prob)

print("ACCURACY: " + str(acc) + "\n")
print("LOG LOSS: " + str(log_loss_score) + "\n")

#========== DATA VISUALIZATION 2D ===========
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Principal Component 1', fontsize=15)
ax.set_ylabel('Principal Component 2', fontsize=15)
ax.set_title('2 Component PCA', fontsize=20)


targets = [1, 0]
colors = ['r', 'g']
for target, color in zip(targets, colors):
    indicesToKeep = finalDf['Target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c=color
               , s=50)
ax.legend(targets)
ax.grid()
plt.show()

#=========== LOGISTIC REGRESSION WITH PCA 3D ============
X = brain_tumor_data.drop("Target", axis=1)
ss = StandardScaler()
X = ss.fit_transform(X)

pca = PCA(n_components=3)
principal_components = pca.fit_transform(X)
principalDF = pd.DataFrame(data=principal_components,
             columns=['PCA1', 'PCA2', 'PCA3'])

finalDf = pd.concat([principalDF, brain_tumor_data[['Target']]], axis=1)

X = finalDf.drop("Target", axis=1)
Y = finalDf["Target"].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)


log_reg.fit(X_train, Y_train)
Y_pred = log_reg.predict(X_test)
Y_pred_prob = log_reg.predict_proba(X_test)

acc = accuracy_score(Y_test, Y_pred)
log_loss_score = log_loss(Y_test, Y_pred_prob)

print("ACCURACY: " + str(acc) + "\n")
print("LOG LOSS: " + str(log_loss_score) + "\n")

#=========== DATA VISUALIZATION 3D ================
fig = plt.figure()
finalDf['Target'] = pd.Categorical(finalDf['Target'])
my_color = finalDf['Target'].cat.codes
ax = Axes3D(fig)
ax.scatter(finalDf['PCA1'], finalDf['PCA2'], finalDf['PCA3'], c=my_color, cmap="Set2_r", s=60)

# make simple, bare axis lines through space:
xAxisLine = ((min(finalDf['PCA1']), max(finalDf['PCA1'])), (0, 0), (0, 0))
ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'r')
yAxisLine = ((0, 0), (min(finalDf['PCA2']), max(finalDf['PCA2'])), (0, 0))
ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'r')
zAxisLine = ((0, 0), (0, 0), (min(finalDf['PCA3']), max(finalDf['PCA3'])))
ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'r')

# label the axes
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title("PCA on the iris data set")
plt.show()

#========== ==============
pca = PCA(.95)
X = brain_tumor_data.drop("Target", axis=1).values
Y = brain_tumor_data["Target"].values
X = ss.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

log_reg.fit(X_train, Y_train)
Y_pred = log_reg.predict(X_test)
Y_pred_prob = log_reg.predict_proba(X_test)

acc = accuracy_score(Y_test, Y_pred)
log_loss_score = log_loss(Y_test, Y_pred_prob)

print("ACCURACY: " + str(acc) + "\n")
print("LOG LOSS: " + str(log_loss_score) + "\n")