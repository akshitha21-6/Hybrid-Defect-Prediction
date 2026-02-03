import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
import math

def fs_none(X, y, names):
    return X, list(names)

def fs_rfe_rf(X, y, names):
    k = max(3, int(math.sqrt(X.shape[1])))
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    rfe = RFE(model, n_features_to_select=k)
    X_new = rfe.fit_transform(X, y)
    selected = np.array(names)[rfe.support_]
    return X_new, list(selected)

def fs_ga(X, y, names):
    k = max(3, int(0.4 * X.shape[1]))
    scores = np.var(X, axis=0)
    idx = np.argsort(scores)[-k:]
    return X[:, idx], list(np.array(names)[idx])

def fs_aco(X, y, names):
    k = max(3, int(0.3 * X.shape[1]))
    scores = np.mean(X, axis=0)
    idx = np.argsort(scores)[-k:]
    return X[:, idx], list(np.array(names)[idx])

def fs_pso(X, y, names):
    k = max(3, int(0.35 * X.shape[1]))
    scores = np.std(X, axis=0)
    idx = np.argsort(scores)[-k:]
    return X[:, idx], list(np.array(names)[idx])

def fs_hybrid(X, y, names):
    k = max(3, int(0.25 * X.shape[1]))
    combined = np.var(X, axis=0) + np.std(X, axis=0)
    idx = np.argsort(combined)[-k:]
    return X[:, idx], list(np.array(names)[idx])
