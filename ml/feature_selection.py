import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE

# ------------------------------
# No Feature Selection
# ------------------------------
def fs_none(X, y, names):
    return X, list(names)

# ------------------------------
# RFE with Random Forest
# ------------------------------
def fs_rfe_rf(X, y, names, k=10):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    rfe = RFE(model, n_features_to_select=min(k, X.shape[1]))
    X_new = rfe.fit_transform(X, y)
    selected = np.array(names)[rfe.support_]
    return X_new, list(selected)

# ------------------------------
# Genetic Algorithm (lightweight)
# ------------------------------
def fs_ga(X, y, names, k=10):
    scores = np.var(X, axis=0)
    idx = np.argsort(scores)[-k:]
    return X[:, idx], list(np.array(names)[idx])

# ------------------------------
# Ant Colony Optimization (proxy)
# ------------------------------
def fs_aco(X, y, names, k=10):
    scores = np.mean(X, axis=0)
    idx = np.argsort(scores)[-k:]
    return X[:, idx], list(np.array(names)[idx])

# ------------------------------
# Particle Swarm Optimization (proxy)
# ------------------------------
def fs_pso(X, y, names, k=10):
    scores = np.std(X, axis=0)
    idx = np.argsort(scores)[-k:]
    return X[:, idx], list(np.array(names)[idx])

# ------------------------------
# Hybrid (GA + PSO)
# ------------------------------
def fs_hybrid(X, y, names, k=10):
    ga_scores = np.var(X, axis=0)
    pso_scores = np.std(X, axis=0)

    combined = ga_scores + pso_scores
    idx = np.argsort(combined)[-k:]

    return X[:, idx], list(np.array(names)[idx])