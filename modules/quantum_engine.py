"""
QUANTUM EYE — Quantum Machine Learning Laboratory Engine
PennyLane-powered Quantum Kernel Support Vector Machine (QSVM) and
Variational Quantum Classifier (VQC) with telemetry, circuit metrics,
and transparent classical-quantum comparison.
"""
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False


class QuantumEngine:
    """Scientific Quantum Machine Learning Engine."""

    def __init__(self, n_qubits: int = 3, backend_name: str = "default.qubit"):
        self.n_qubits = n_qubits
        self.backend_name = backend_name
        self.is_simulator = True
        self.dev = None
        self._init_device()

    def _init_device(self):
        if not PENNYLANE_AVAILABLE:
            return
        try:
            self.dev = qml.device(self.backend_name, wires=self.n_qubits)
        except Exception:
            # Fallback to default.qubit
            self.backend_name = "default.qubit"
            self.dev = qml.device("default.qubit", wires=self.n_qubits)

    def _quantum_feature_map(self, x: np.ndarray):
        """
        ZZ Feature Map:
        Hadamard layer -> Rz rotations based on features -> CNOT entanglement -> Rz based on feature products.
        """
        for i in range(self.n_qubits):
            qml.Hadamard(wires=i)
            qml.RZ(2.0 * x[i], wires=i)

        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
            qml.RZ(2.0 * (np.pi - x[i]) * (np.pi - x[i + 1]), wires=i + 1)
            qml.CNOT(wires=[i, i + 1])

    def compute_kernel_matrix(self, X1: np.ndarray, X2: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Calculates Quantum Kernel Gram Matrix:
        K(x1, x2) = |<phi(x1) | phi(x2)>|^2
        Returns (kernel_matrix, telemetry)
        """
        if not PENNYLANE_AVAILABLE or self.dev is None:
            raise RuntimeError("PennyLane quantum backend is not available.")

        n1, n2 = len(X1), len(X2)
        kernel_matrix = np.zeros((n1, n2))

        # Define quantum kernel circuit for two feature vectors
        @qml.qnode(self.dev)
        def kernel_circuit(x1, x2):
            self._quantum_feature_map(x1)
            qml.adjoint(self._quantum_feature_map)(x2)
            return qml.probs(wires=range(self.n_qubits))

        circuit_evaluations = 0
        t0 = time.perf_counter()

        is_symmetric = (X1 is X2) or (n1 == n2 and np.array_equal(X1, X2))

        for i in range(n1):
            start_j = i if is_symmetric else 0
            for j in range(start_j, n2):
                if is_symmetric and i == j:
                    kernel_matrix[i, j] = 1.0
                else:
                    # Probability of measuring all qubits in |0...0> state
                    probs = kernel_circuit(X1[i], X2[j])
                    circuit_evaluations += 1
                    fidelity = float(probs[0])
                    kernel_matrix[i, j] = fidelity
                    if is_symmetric:
                        kernel_matrix[j, i] = fidelity

        total_time = time.perf_counter() - t0

        telemetry = {
            "backend": self.backend_name,
            "is_simulator": self.is_simulator,
            "n_qubits": self.n_qubits,
            "circuit_depth": 2 * (1 + 1 + 3 * (self.n_qubits - 1)),  # estimated depth of ZZ map + adjoint
            "circuit_evaluations": circuit_evaluations,
            "total_execution_time_sec": total_time,
            "time_per_evaluation_ms": (total_time / max(1, circuit_evaluations)) * 1000.0
        }

        return kernel_matrix, telemetry

    def train_quantum_kernel_svm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        C: float = 1.0
    ) -> Dict[str, Any]:
        """
        Full Quantum Kernel SVM workflow:
        1. Dimension reduction to n_qubits using PCA.
        2. Feature scaling to [0, pi] for angle embedding.
        3. Quantum Kernel Matrix generation.
        4. Classical SVM with precomputed quantum kernel.
        5. Baseline Classical SVM comparison.
        """
        # Step 1: PCA reduction to n_qubits
        pca = PCA(n_components=self.n_qubits)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        # Step 2: Scale into [0, pi]
        scaler = MinMaxScaler(feature_range=(0, np.pi))
        X_train_q = scaler.fit_transform(X_train_pca)
        X_test_q = scaler.transform(X_test_pca)

        # Step 3: Compute Quantum Train and Test Kernels
        K_train, train_telemetry = self.compute_kernel_matrix(X_train_q, X_train_q)
        K_test, test_telemetry = self.compute_kernel_matrix(X_test_q, X_train_q)

        # Step 4: Fit precomputed SVM
        t_fit_start = time.perf_counter()
        qsvm = SVC(kernel="precomputed", C=C)
        qsvm.fit(K_train, y_train)
        fit_time = time.perf_counter() - t_fit_start

        # Inference
        t_inf_start = time.perf_counter()
        y_pred = qsvm.predict(K_test)
        inference_time = time.perf_counter() - t_inf_start

        # Quantum Metrics
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
        rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
        cm = confusion_matrix(y_test, y_pred)

        # Step 5: Classical SVM Baseline on same reduced data for fair comparison
        t_base_train_start = time.perf_counter()
        classical_svm = SVC(kernel="rbf", C=C)
        classical_svm.fit(X_train_q, y_train)
        classical_train_time = time.perf_counter() - t_base_train_start

        t_base_inf_start = time.perf_counter()
        y_pred_classical = classical_svm.predict(X_test_q)
        classical_inf_time = time.perf_counter() - t_base_inf_start

        c_acc = float(accuracy_score(y_test, y_pred_classical))
        c_f1 = float(f1_score(y_test, y_pred_classical, average="weighted", zero_division=0))

        # Quantum processing overhead
        q_overhead_ratio = (train_telemetry["total_execution_time_sec"] + fit_time) / max(1e-6, classical_train_time)

        return {
            "quantum": {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "confusion_matrix": cm,
                "y_pred": y_pred,
                "y_test": y_test,
                "train_telemetry": train_telemetry,
                "test_telemetry": test_telemetry,
                "kernel_train_time": train_telemetry["total_execution_time_sec"],
                "svm_fit_time": fit_time,
                "inference_time": inference_time,
                "total_time": train_telemetry["total_execution_time_sec"] + test_telemetry["total_execution_time_sec"] + fit_time + inference_time,
                "K_train": K_train,
                "K_test": K_test
            },
            "classical_baseline": {
                "accuracy": c_acc,
                "f1_score": c_f1,
                "train_time": classical_train_time,
                "inference_time": classical_inf_time,
                "y_pred": y_pred_classical
            },
            "telemetry": {
                "backend": self.backend_name,
                "n_qubits": self.n_qubits,
                "is_simulator": self.is_simulator,
                "quantum_overhead_ratio": q_overhead_ratio,
                "pca_variance_ratio": pca.explained_variance_ratio_.tolist(),
                "total_pca_variance": float(np.sum(pca.explained_variance_ratio_))
            }
        }
