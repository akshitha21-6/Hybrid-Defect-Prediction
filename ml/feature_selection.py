import numpy as np
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE

# =====================================================
# NO FEATURE SELECTION (ALL FEATURES)
# =====================================================
def fs_none(X, y, feature_names):
    return X, list(feature_names)

# =====================================================
# RFE + RANDOM FOREST (√N FEATURES)
# =====================================================
def fs_rfe_rf(X, y, feature_names):
    k = max(3, int(math.sqrt(X.shape[1])))

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    rfe = RFE(
        estimator=model,
        n_features_to_select=k
    )

    X_new = rfe.fit_transform(X, y)
    selected = np.array(feature_names)[rfe.support_]

    return X_new, list(selected)

# =====================================================
# GENETIC ALGORITHM (PROXY – TOP 40%)
# =====================================================
def fs_ga(X, y, feature_names):
    k = max(3, int(0.40 * X.shape[1]))

    scores = np.var(X, axis=0)
    idx = np.argsort(scores)[-k:]

    return X[:, idx], list(np.array(feature_names)[idx])

# =====================================================
# ANT COLONY OPTIMIZATION (PROXY – TOP 30%)
# =====================================================
def fs_aco(X, y, feature_names):
    k = max(3, int(0.30 * X.shape[1]))

    scores = np.mean(X, axis=0)
    idx = np.argsort(scores)[-k:]

    return X[:, idx], list(np.array(feature_names)[idx])

# =====================================================
# PARTICLE SWARM OPTIMIZATION (PROXY – TOP 35%)
# =====================================================
def fs_pso(X, y, feature_names):
    k = max(3, int(0.35 * X.shape[1]))

    scores = np.std(X, axis=0)
    idx = np.argsort(scores)[-k:]

    return X[:, idx], list(np.array(feature_names)[idx])

# =====================================================
# HYBRID GA + PSO (PROXY – TOP 25%)
# =====================================================
def fs_hybrid(X, y, feature_names):
    k = max(3, int(0.25 * X.shape[1]))

    ga_scores = np.var(X, axis=0)
    pso_scores = np.std(X, axis=0)

    combined = ga_scores + pso_scores
    idx = np.argsort(combined)[-k:]

    return X[:, idx], list(np.array(feature_names)[idx])
