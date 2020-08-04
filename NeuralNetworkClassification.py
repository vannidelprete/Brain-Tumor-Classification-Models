#======== IMPORT =============
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import warnings

warnings.filterwarnings("ignore")
#========= READ DATASET =======
brain_tumor_data = pd.read_csv(r"Data/bt_dataset_t3_fixed.csv")
del brain_tumor_data["Unnamed: 0"]

#========= SPLIT TEST TRAIN =========
X = brain_tumor_data.drop("Target", axis=1).values
Y = brain_tumor_data["Target"].values
ss = StandardScaler()
X = ss.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

#========= MULTI-LAYER PERCEPTRON CLASSIFICATION ============
mlp = MLPClassifier(hidden_layer_sizes=(100,), verbose=True, max_iter=300)
mlp.fit(X_train, Y_train)

y_pred_train = mlp.predict(X_train)
y_prob_train = mlp.predict_proba(X_train)

y_pred = mlp.predict(X_test)
y_prob = mlp.predict_proba(X_test)

accuracy_train = accuracy_score(Y_train, y_pred_train)

loss_train = log_loss(Y_train, y_prob_train)

print("ACCURACY: TRAIN=%.4f" % accuracy_train)
print("LOG LOSS: TRAIN=%.4f" % loss_train)