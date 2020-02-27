#======== IMPORT =============
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
import matplotlib.pyplot as plt
import seaborn as sns

#========= READ DATASET =======
brain_tumor_data = pd.read_csv(r"Data/bt_dataset_t3_fixed.csv")
del brain_tumor_data["Unnamed: 0"]
#========= PLOTTING HEATMAP ==========
sns.heatmap(brain_tumor_data.corr(), xticklabels=brain_tumor_data.columns, yticklabels=brain_tumor_data.columns)
plt.show()

#========= FEATURE SELECTED ==========
feats = ["Mean", "Entropy", "Skewness", "Kurtosis", "Energy", "ASM", "Homogeneity", "SSIM"]
sns.heatmap(brain_tumor_data[feats].corr(), xticklabels=brain_tumor_data[feats].columns, yticklabels=brain_tumor_data[feats].columns)
plt.show()

feats = ["Mean", "Entropy", "Skewness", "Kurtosis", "PSNR", "Dissimilarity", "DC", "SSIM"]
sns.heatmap(brain_tumor_data[feats].corr(), xticklabels=brain_tumor_data[feats].columns, yticklabels=brain_tumor_data[feats].columns)
plt.show()

sns.pairplot(brain_tumor_data[feats])
plt.show()

#======== LOGISTIC REGRESSION =========
X = brain_tumor_data[feats]
Y = brain_tumor_data["Target"].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

log_reg = LogisticRegression()
log_reg.fit(X_train, Y_train)
Y_pred = log_reg.predict(X_test)
Y_pred_prob = log_reg.predict_proba(X_test)

#========== COMPUTE METRICS ==============
acc = accuracy_score(Y_test, Y_pred)
log_loss_score = log_loss(Y_test, Y_pred_prob)

print("ACCURACY: " + str(acc) + "\n")
print("LOG LOSS: " + str(log_loss_score) + "\n")