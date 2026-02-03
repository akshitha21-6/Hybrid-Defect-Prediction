import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
import math

# ------------------------------
# No Feature Selection
# ------------------------------
def fs_none(X, y, names):
    return X, list(names)

# ------------------------------
# RFE with Random Forest (√N)
# ------------------------------
def fs_rfe_rf(X, y, names):
    k = max(3, int(math.sqrt(X.shape[1])))
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    rfe = RFE(model, n_features_to_select=k)
    X_new = rfe.fit_transform(X, y)
    selected = np.array(names)[rfe.support_]
    return X_new, list(selected)

# ------------------------------
# Genetic Algorithm (proxy – 40%)
# ------------------------------
def fs_ga(X, y, names):
    k = max(3, int(0.4 * X.shape[1]))
    scores = np.var(X, axis=0)
    idx = np.argsort(scores)[-k:]
    return X[:, idx], list(np.array(names)[idx])

# ------------------------------
# Ant Colony Optimization (proxy – 30%)
# ------------------------------
def fs_aco(X, y, names):
    k = max(3, int(0.3 * X.shape[1]))
    scores = np.mean(X, axis=0)
    idx = np.argsort(scores)[-k:]
    return X[:, idx], list(np.array(names)[idx])

# ------------------------------
# Particle Swarm Optimization (proxy – 35%)
# ------------------------------
def fs_pso(X, y, names):
    k = max(3, int(0.35 * X.shape[1]))
    scores = np.std(X, axis=0)
    idx = np.argsort(scores)[-k:]
    return X[:, idx], list(np.array(names)[idx])

# ------------------------------
# Hybrid GA + PSO (proxy – 25%)
# ------------------------------
def fs_hybrid(X, y, names):
    k = max(3, int(0.25 * X.shape[1]))
    ga_scores = np.var(X, axis=0)
    pso_scores = np.std(X, axis=0)
    combined = ga_scores + pso_scores
    idx = np.argsort(combined)[-k:]
    return X[:, idx], list(np.array(names)[idx])
