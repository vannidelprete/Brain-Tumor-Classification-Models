#======== IMPORT =============
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


#========= READ DATASET =======
brain_tumor_data = pd.read_csv(r"Data/bt_dataset_t3_fixed.csv")
del brain_tumor_data["Unnamed: 0"]

#========= KNN CLASSIFIER ===========
X = brain_tumor_data.drop("Target", axis=1).values
Y = brain_tumor_data["Target"].values
ss = StandardScaler()
X = ss.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
print(str(knn.fit(X_train, Y_train))+"\n")


Y_pred = knn.predict(X_test)
Y_prob = knn.predict_proba(X_test)


acc = accuracy_score(Y_test, Y_pred)

log_loss_score = log_loss(Y_test, Y_prob)

print("ACCURACY: " + str(acc) + "\n")
print("LOG LOSS: " + str(log_loss_score) + "\n")

#=========== IMPROVE THE MODEL ==========
Ks = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
accuracys = np.array([])
scores = np.array([])
for K in Ks:
    print("============ "+"K="+str(K)+" ============")
    knn = KNeighborsClassifier(n_neighbors=K)
    knn.fit(X_train, Y_train)

    Y_pred_train = knn.predict(X_train)
    Y_prob_train = knn.predict_proba(X_train)

    Y_pred = knn.predict(X_test)
    Y_prob = knn.predict_proba(X_test)
    Y_pred = knn.predict(X_test)
    Y_prob = knn.predict_proba(X_test)

    acc = accuracy_score(Y_test, Y_pred)
    accuracys = np.append(accuracys,acc)
    log_loss_score = log_loss(Y_test, Y_prob)
    scores = np.append(scores, log_loss_score)

    print("ACCURACY: " + str(acc))
    print("LOG LOSS: " + str(log_loss_score) + "\n")
    print("===========================================")


#============= PLOTTING =====================
x_plot = np.array(Ks)
y_plot = accuracys
y_plot_2 = scores

plt.plot(x_plot, y_plot, 'r--', label="Accuracy")
plt.xticks(np.arange(x_plot.min(), x_plot.max(), 1))
plt.legend()
plt.show()

plt.plot(x_plot, y_plot_2, 'bs', label='Log loss score')
plt.xticks(np.arange(x_plot.min(), x_plot.max(), 1))
plt.legend()
plt.show()