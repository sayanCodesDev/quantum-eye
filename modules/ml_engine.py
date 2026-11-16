"""
QUANTUM EYE — Machine Learning Engine
Implements Supervised Classification (RF, SVM, DT, KNN, Gradient Boosting),
Unsupervised Clustering/Dimensionality Reduction (K-Means, DBSCAN, PCA, Isolation Forest),
and the 3-Class Signal Intelligence Experiment (Periodic vs Stochastic vs Chaotic).
Strict train/test leakage prevention: scalers fit ONLY on training sets.
"""
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from modules.signal_generator import SignalGenerator
from modules.feature_extraction import extract_signal_features

class MLEngine:
    """Core classical machine learning laboratory."""

    # ----------------- Supervised Training -----------------
    @staticmethod
    def get_supervised_classifier(model_name: str, **kwargs):
        name = model_name.lower()
        if "random forest" in name or "rf" in name:
            n_est = kwargs.get("n_estimators", 100)
            max_depth = kwargs.get("max_depth", None)
            return RandomForestClassifier(n_estimators=n_est, max_depth=max_depth, random_state=42)
        elif "svm" in name or "support vector" in name:
            kernel = kwargs.get("kernel", "rbf")
            c_val = kwargs.get("C", 1.0)
            return SVC(kernel=kernel, C=c_val, probability=True, random_state=42)
        elif "decision tree" in name or "dt" in name:
            max_depth = kwargs.get("max_depth", None)
            return DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        elif "knn" in name or "k-nearest" in name:
            k = kwargs.get("n_neighbors", 5)
            return KNeighborsClassifier(n_neighbors=k)
        elif "gradient boosting" in name or "gb" in name:
            n_est = kwargs.get("n_estimators", 100)
            lr = kwargs.get("learning_rate", 0.1)
            return GradientBoostingClassifier(n_estimators=n_est, learning_rate=lr, random_state=42)
        else:
            return RandomForestClassifier(n_estimators=100, random_state=42)

    @staticmethod
    def train_and_evaluate_supervised(
        X: np.ndarray,
        y: np.ndarray,
        model_name: str = "Random Forest",
        test_size: float = 0.25,
        cv_folds: int = 5,
        feature_names: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Trains and evaluates classifier with strict leakage-free standard scaling.
        """
        # Ensure numpy arrays with standard numeric indexing
        X = np.asarray(X)
        y = np.asarray(y)

        # Stratified train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
        )

        # Fit scaler ONLY on train set to prevent data leakage!
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        clf = MLEngine.get_supervised_classifier(model_name, **kwargs)

        # Train with execution time measurement
        t0 = time.perf_counter()
        clf.fit(X_train_scaled, y_train)
        train_time = time.perf_counter() - t0

        # Inference timing
        t1 = time.perf_counter()
        y_pred = clf.predict(X_test_scaled)
        inference_time = time.perf_counter() - t1

        # Prediction probabilities if supported
        y_prob = clf.predict_proba(X_test_scaled) if hasattr(clf, "predict_proba") else None

        # Metrics
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
        rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
        cm = confusion_matrix(y_test, y_pred)
        labels = np.unique(np.concatenate([y_train, y_test]))

        # Cross-validation
        cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=min(cv_folds, len(y_train)))
        cv_mean = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))

        return {
            "model_name": model_name,
            "model": clf,
            "scaler": scaler,
            "X_train": X_train_scaled,
            "X_test": X_test_scaled,
            "y_train": y_train,
            "y_test": y_test,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "confusion_matrix": cm,
            "classes": labels,
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "train_time_sec": train_time,
            "inference_time_sec": inference_time,
            "feature_names": feature_names or [f"feat_{i}" for i in range(X.shape[1])]
        }

    # ----------------- Unsupervised Learning -----------------
    @staticmethod
    def run_pca(X: np.ndarray, n_components: int = 3) -> Dict[str, Any]:
        """Performs PCA projection and computes variance explained."""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        n_comp = min(n_components, X.shape[1], X.shape[0])
        pca = PCA(n_components=n_comp)
        X_pca = pca.fit_transform(X_scaled)
        
        return {
            "X_pca": X_pca,
            "pca_model": pca,
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "total_variance_explained": float(np.sum(pca.explained_variance_ratio_))
        }

    @staticmethod
    def run_kmeans(X: np.ndarray, n_clusters: int = 3) -> Dict[str, Any]:
        """Performs K-Means clustering."""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        return {
            "cluster_labels": labels,
            "cluster_centers": km.cluster_centers_,
            "inertia": float(km.inertia_)
        }

    @staticmethod
    def run_dbscan(X: np.ndarray, eps: float = 0.5, min_samples: int = 5) -> Dict[str, Any]:
        """Performs density-based spatial clustering (DBSCAN)."""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(X_scaled)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int(np.sum(labels == -1))
        return {
            "cluster_labels": labels,
            "n_clusters": n_clusters,
            "n_noise": n_noise
        }

    # ----------------- 3-Class Controlled Signal Experiment -----------------
    @staticmethod
    def run_signal_intelligence_experiment(
        n_signals_per_class: int = 25,
        signal_length: int = 500,
        fs: float = 100.0,
        noise_snr_db: Optional[float] = 20.0
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Creates an independent controlled experiment dataset for:
        1. Periodic signals (Sine, Square, Sawtooth, Composite)
        2. Stochastic signals (Gaussian white noise, Pink noise, Brown noise)
        3. Deterministic Chaotic signals (Lorenz, Logistic, Rössler)
        
        Each signal is independently generated to ensure zero window-leakage.
        """
        features_list = []
        labels_list = []

        # 1. Periodic Class
        for i in range(n_signals_per_class):
            sig_type = i % 4
            freq = np.random.uniform(2.0, 15.0)
            if sig_type == 0:
                _, s = SignalGenerator.generate_sine(n_samples=signal_length, fs=fs, freq=freq)
            elif sig_type == 1:
                _, s = SignalGenerator.generate_square(n_samples=signal_length, fs=fs, freq=freq)
            elif sig_type == 2:
                _, s = SignalGenerator.generate_sawtooth(n_samples=signal_length, fs=fs, freq=freq)
            else:
                _, s = SignalGenerator.generate_composite(n_samples=signal_length, fs=fs, base_freq=freq)
                
            if noise_snr_db is not None:
                s = SignalGenerator.inject_noise_snr(s, snr_db=noise_snr_db)
            
            f = extract_signal_features(s, fs=fs)
            features_list.append(f)
            labels_list.append("Periodic")

        # 2. Stochastic Class
        for i in range(n_signals_per_class):
            sig_type = i % 3
            if sig_type == 0:
                _, s = SignalGenerator.generate_gaussian_noise(n_samples=signal_length, fs=fs)
            elif sig_type == 1:
                _, s = SignalGenerator.generate_colored_noise(n_samples=signal_length, fs=fs, color="pink")
            else:
                _, s = SignalGenerator.generate_colored_noise(n_samples=signal_length, fs=fs, color="brown")
                
            f = extract_signal_features(s, fs=fs)
            features_list.append(f)
            labels_list.append("Stochastic")

        # 3. Chaotic Class
        for i in range(n_signals_per_class):
            sig_type = i % 3
            if sig_type == 0:
                rho = np.random.uniform(24.0, 32.0)
                _, s = SignalGenerator.generate_lorenz(n_samples=signal_length, rho=rho)
            elif sig_type == 1:
                r = np.random.uniform(3.75, 3.98)
                _, s = SignalGenerator.generate_logistic_map(n_samples=signal_length, r=r)
            else:
                c = np.random.uniform(5.0, 6.0)
                _, s = SignalGenerator.generate_rossler(n_samples=signal_length, c=c)
                
            if noise_snr_db is not None:
                s = SignalGenerator.inject_noise_snr(s, snr_db=noise_snr_db)
                
            f = extract_signal_features(s, fs=fs)
            features_list.append(f)
            labels_list.append("Chaotic")

        X_df = pd.DataFrame(features_list)
        y_ser = pd.Series(labels_list, name="signal_class")
        return X_df, y_ser
