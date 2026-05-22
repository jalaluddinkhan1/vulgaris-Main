import numpy as np
import time
import json
from typing import Dict, Callable, List, Optional
from dataclasses import dataclass, field


@dataclass
class BenchmarkResult:
    dataset_name: str
    task: str
    mae: float = 0.0
    rmse: float = 0.0
    mape: float = 0.0
    accuracy: float = 0.0
    f1: float = 0.0
    coverage_90: float = 0.0
    coverage_95: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p99_ms: float = 0.0
    memory_mb: float = 0.0
    n_params: int = 0
    flops_per_step: int = 0


class SyntheticDataset:
    """High-fidelity synthetic industrial telemetry generator."""

    @staticmethod
    def power_grid(n_samples: int = 10000, n_sensors: int = 32,
                   sample_rate: float = 60.0, add_faults: bool = True,
                   add_noise: bool = True) -> tuple:
        window_size = 128
        rng = np.random.default_rng(42)
        t = np.arange(window_size) / sample_rate  # time vector per window

        # Base frequency components (50 Hz and harmonics)
        X = np.zeros((n_samples, n_sensors, window_size), dtype=np.float32)
        y = np.zeros((n_samples, 4), dtype=np.float32)  # [voltage_norm, freq_dev, fault_prob, load]

        for i in range(n_samples):
            # Slow load drift
            load = 0.5 + 0.4 * np.sin(2 * np.pi * i / 1000.0)
            freq_dev = rng.normal(0.0, 0.02)  # Hz deviation from nominal

            for s in range(n_sensors):
                phase = 2 * np.pi * s / n_sensors
                # Fundamental + 3rd harmonic + 5th harmonic
                sig = (
                    load * np.sin(2 * np.pi * (50.0 + freq_dev) * t + phase)
                    + 0.1 * np.sin(2 * np.pi * (150.0 + 3 * freq_dev) * t + phase)
                    + 0.05 * np.sin(2 * np.pi * (250.0 + 5 * freq_dev) * t + phase)
                )
                if add_noise:
                    sig += rng.normal(0.0, 0.02, size=window_size)
                X[i, s, :] = sig.astype(np.float32)

            fault_prob = 0.0
            if add_faults and rng.random() < 0.05:
                # Voltage sag event
                fault_start = rng.integers(0, window_size // 2)
                fault_len = rng.integers(10, 30)
                fault_end = min(fault_start + fault_len, window_size)
                X[i, :, fault_start:fault_end] *= rng.uniform(0.5, 0.85)
                fault_prob = 1.0

            y[i] = [load, freq_dev, fault_prob, load * (1.0 + freq_dev)]

        sensor_names = [f'sensor_{s}' for s in range(n_sensors)]
        fault_times = np.where(y[:, 2] > 0.5)[0].tolist()
        metadata = {
            'sensor_names': sensor_names,
            'fault_times': fault_times,
            'sample_rate': sample_rate,
            'window_size': window_size,
        }
        return X, y, metadata

    @staticmethod
    def industrial_process(n_samples: int = 10000, n_sensors: int = 24,
                           regime_changes: bool = True) -> tuple:
        window_size = 64
        rng = np.random.default_rng(123)
        X = np.zeros((n_samples, n_sensors, window_size), dtype=np.float32)
        y = np.zeros((n_samples, 3), dtype=np.float32)  # [temp_pred, pressure_pred, anomaly]

        # Define 3 production regimes
        regime_temps = [300.0, 450.0, 200.0]
        regime_pressures = [5.0, 12.0, 2.5]

        regime = 0
        for i in range(n_samples):
            if regime_changes and i % 2000 == 0 and i > 0:
                regime = (regime + 1) % 3

            base_temp = regime_temps[regime]
            base_pressure = regime_pressures[regime]
            drift = 0.001 * (i % 2000)  # gradual drift within regime

            t = np.linspace(0, 1, window_size)
            for s in range(n_sensors):
                sensor_type = s % 3  # 0=temp, 1=pressure, 2=flow
                if sensor_type == 0:
                    sig = (base_temp + drift + 5.0 * np.sin(2 * np.pi * t)
                           + rng.normal(0, 2.0, window_size))
                elif sensor_type == 1:
                    sig = (base_pressure + 0.1 * drift + 0.5 * np.cos(2 * np.pi * t)
                           + rng.normal(0, 0.2, window_size))
                else:
                    sig = (10.0 + 2.0 * np.sin(4 * np.pi * t + s)
                           + rng.normal(0, 0.5, window_size))
                X[i, s, :] = sig.astype(np.float32)

            anomaly = 0.0
            if rng.random() < 0.03:
                # Sudden spike
                X[i, rng.integers(0, n_sensors), rng.integers(0, window_size)] *= 3.0
                anomaly = 1.0

            y[i] = [base_temp + drift, base_pressure, anomaly]

        metadata = {
            'regime_temps': regime_temps,
            'regime_pressures': regime_pressures,
            'window_size': window_size,
            'n_regimes': 3,
        }
        return X, y, metadata

    @staticmethod
    def telecom_ran(n_samples: int = 10000, n_cells: int = 16,
                    add_anomalies: bool = True) -> tuple:
        window_size = 60  # 1-minute windows at 1s resolution
        rng = np.random.default_rng(7)
        # KPIs: PRB_util, SINR, throughput, latency per cell = n_cells * 4
        n_sensors = n_cells * 4
        X = np.zeros((n_samples, n_sensors, window_size), dtype=np.float32)
        y = np.zeros((n_samples, 2), dtype=np.float32)  # [avg_throughput, anomaly_score]

        for i in range(n_samples):
            hour_of_day = (i // 60) % 24
            # Traffic pattern: rush hours
            load_factor = 0.3 + 0.6 * np.exp(-0.5 * ((hour_of_day - 8) ** 2) / 4.0)
            load_factor += 0.4 * np.exp(-0.5 * ((hour_of_day - 18) ** 2) / 4.0)
            load_factor = float(np.clip(load_factor, 0.1, 1.0))

            t = np.arange(window_size)
            total_throughput = 0.0

            for c in range(n_cells):
                base_idx = c * 4
                prb = load_factor * (70 + 20 * rng.random()) + 5 * np.sin(2 * np.pi * t / 60.0)
                sinr = 15.0 + 5.0 * rng.random() - 2.0 * load_factor + rng.normal(0, 1.0, window_size)
                throughput = load_factor * 100.0 + 10 * rng.random(window_size)
                latency = 5.0 + 2.0 * load_factor + rng.exponential(1.0, window_size)

                if add_anomalies and rng.random() < 0.02:
                    drop_start = rng.integers(0, window_size // 2)
                    throughput[drop_start:drop_start + 10] *= 0.1
                    sinr[drop_start:drop_start + 10] -= 10.0

                X[i, base_idx, :] = np.clip(prb, 0, 100).astype(np.float32)
                X[i, base_idx + 1, :] = np.clip(sinr, -5, 40).astype(np.float32)
                X[i, base_idx + 2, :] = np.clip(throughput, 0, 200).astype(np.float32)
                X[i, base_idx + 3, :] = np.clip(latency, 1, 100).astype(np.float32)
                total_throughput += float(throughput.mean())

            anomaly_score = float(np.std(X[i, :, :]))
            y[i] = [total_throughput / n_cells, anomaly_score]

        metadata = {
            'n_cells': n_cells,
            'kpis': ['prb_util', 'sinr', 'throughput', 'latency'],
            'window_size': window_size,
            'sample_rate': 1.0,
        }
        return X, y, metadata

    @staticmethod
    def predictive_maintenance(n_samples: int = 10000, n_sensors: int = 20,
                               failure_rate: float = 0.05) -> tuple:
        window_size = 128
        rng = np.random.default_rng(99)
        X = np.zeros((n_samples, n_sensors, window_size), dtype=np.float32)
        y = np.zeros((n_samples, 2), dtype=np.float32)  # [rul, failure_within_100]

        # Bearing degradation stages: 0=healthy, 1=degrading, 2=critical, 3=failed
        machine_health = np.ones(n_samples, dtype=np.float32)
        rul = np.zeros(n_samples, dtype=np.float32)

        # Generate degradation trajectory
        lifespan = int(n_samples * (1.0 - failure_rate))
        for i in range(n_samples):
            progress = i / lifespan
            health = np.clip(1.0 - progress ** 2, 0.0, 1.0)
            machine_health[i] = health
            rul[i] = max(0.0, (lifespan - i) / lifespan * 100.0)

        t = np.linspace(0, 1, window_size)
        for i in range(n_samples):
            health = machine_health[i]
            degradation = 1.0 - health

            for s in range(n_sensors):
                if s < 8:  # vibration sensors
                    base_freq = 30.0 + s * 10.0
                    amplitude = 1.0 + 3.0 * degradation
                    # Bearing fault frequencies appear with degradation
                    fault_amp = degradation * 2.0
                    sig = (amplitude * np.sin(2 * np.pi * base_freq * t)
                           + fault_amp * np.sin(2 * np.pi * 4.7 * base_freq * t)  # BPFO
                           + rng.normal(0, 0.1 + 0.5 * degradation, window_size))
                elif s < 14:  # temperature sensors
                    base_temp = 60.0 + degradation * 40.0
                    sig = (base_temp + 5.0 * np.sin(2 * np.pi * 0.1 * t)
                           + rng.normal(0, 1.0 + 2.0 * degradation, window_size))
                else:  # current sensors
                    base_current = 5.0 + degradation * 3.0
                    sig = (base_current + 0.5 * np.sin(2 * np.pi * 50.0 * t)
                           + rng.normal(0, 0.2 + degradation, window_size))
                X[i, s, :] = sig.astype(np.float32)

            failure_within_100 = 1.0 if rul[i] < 100.0 and rul[i] >= 0.0 else 0.0
            y[i] = [rul[i], failure_within_100]

        metadata = {
            'n_sensors': n_sensors,
            'sensor_types': ['vibration'] * 8 + ['temperature'] * 6 + ['current'] * 6,
            'rul_max': 100.0,
            'window_size': window_size,
            'lifespan': lifespan,
        }
        return X, y, metadata


class VulgarisBenchmark:
    def __init__(self, model, verbose: bool = True):
        self.model = model
        self.verbose = verbose
        self.results: List[BenchmarkResult] = []

    def run_all(self) -> List[BenchmarkResult]:
        datasets = [
            ('power_grid', SyntheticDataset.power_grid, 'regression', 0),
            ('industrial_process', SyntheticDataset.industrial_process, 'regression', 1),
            ('telecom_ran', SyntheticDataset.telecom_ran, 'regression', 2),
            ('predictive_maintenance', SyntheticDataset.predictive_maintenance, 'regression', 3),
        ]
        self.results = []
        for name, generator, task, domain_idx in datasets:
            if self.verbose:
                print(f"Running benchmark: {name} ...")
            try:
                X, y, metadata = generator()
                result = self.run_dataset(name, X, y, task=task, domain_idx=domain_idx)
                self.results.append(result)
                if self.verbose:
                    print(f"  {name}: MAE={result.mae:.4f}, RMSE={result.rmse:.4f}, "
                          f"P50={result.latency_p50_ms:.2f}ms, P99={result.latency_p99_ms:.2f}ms")
            except Exception as e:
                if self.verbose:
                    print(f"  {name}: FAILED — {e}")
                self.results.append(BenchmarkResult(dataset_name=name, task=task))
        return self.results

    def run_dataset(self, name: str, X: np.ndarray, y: np.ndarray,
                    task: str = 'regression', domain_idx: int = 0) -> BenchmarkResult:
        from inference.streaming import StreamingInference

        n = len(X)
        n_train = int(n * 0.6)
        n_val = int(n * 0.2)
        X_train, y_train = X[:n_train], y[:n_train]
        X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
        X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]

        engine = StreamingInference(
            model=self.model,
            batch_size=1,
            normalize_input=True,
            mode='predictive',
        )
        engine.set_domain(domain_idx)

        # Warm up on training set (up to 100 steps, using last timestep of each window)
        n_warmup_steps = min(100, n_train)
        for i in range(n_warmup_steps):
            x_t = X_train[i, :, -1].astype(np.float64)
            engine.step(x_t, domain_idx=domain_idx)

        # Evaluate on test set
        preds = []
        engine.reset_state()
        for i in range(len(X_test)):
            x_t = X_test[i, :, -1].astype(np.float64)
            result = engine.step(x_t, domain_idx=domain_idx)
            pred = result['prediction'].flatten()
            preds.append(pred)

        preds = np.array(preds)
        # Trim predictions to match y_test columns if needed
        min_cols = min(preds.shape[1], y_test.shape[1])
        preds_trim = preds[:, :min_cols]
        y_trim = y_test[:, :min_cols]

        # Latency stats
        latency_stats = self.measure_latency(X_test[:min(100, len(X_test))], n_warmup=5, n_measure=50)

        # Memory estimate
        n_params = self.model.n_base_params() + self.model.n_adapter_params()
        memory_mb = n_params * 8 / (1024 ** 2)  # float64

        if task == 'regression':
            metrics = self._compute_regression_metrics(preds_trim, y_trim)
            result = BenchmarkResult(
                dataset_name=name,
                task=task,
                mae=metrics['mae'],
                rmse=metrics['rmse'],
                mape=metrics['mape'],
                coverage_90=self._conformal_coverage(preds_trim, y_trim, alpha=0.10),
                coverage_95=self._conformal_coverage(preds_trim, y_trim, alpha=0.05),
                latency_p50_ms=latency_stats['p50'],
                latency_p99_ms=latency_stats['p99'],
                memory_mb=memory_mb,
                n_params=n_params,
                flops_per_step=self._estimate_flops(),
            )
        else:
            # Binarize for classification metrics
            y_bin = (y_trim[:, 0] > 0.5).astype(int)
            p_bin = (preds_trim[:, 0] > 0.5).astype(int)
            metrics = self._compute_classification_metrics(p_bin, y_bin)
            result = BenchmarkResult(
                dataset_name=name,
                task=task,
                accuracy=metrics['accuracy'],
                f1=metrics['f1'],
                latency_p50_ms=latency_stats['p50'],
                latency_p99_ms=latency_stats['p99'],
                memory_mb=memory_mb,
                n_params=n_params,
                flops_per_step=self._estimate_flops(),
            )
        return result

    def measure_latency(self, X: np.ndarray, n_warmup: int = 10,
                        n_measure: int = 100) -> dict:
        from inference.streaming import StreamingInference
        engine = StreamingInference(
            model=self.model, batch_size=1, normalize_input=False, mode='predictive',
        )
        latencies = []
        n = min(n_warmup + n_measure, len(X))

        for i in range(min(n_warmup, len(X))):
            x_t = X[i, :, -1].astype(np.float64)
            engine.step(x_t)

        for i in range(n_warmup, min(n_warmup + n_measure, len(X))):
            x_t = X[i, :, -1].astype(np.float64)
            t0 = time.perf_counter()
            engine.step(x_t)
            dt = (time.perf_counter() - t0) * 1000.0
            latencies.append(dt)

        if not latencies:
            return {'p50': 0.0, 'p95': 0.0, 'p99': 0.0, 'mean': 0.0, 'max': 0.0}

        arr = np.array(latencies)
        return {
            'p50': float(np.percentile(arr, 50)),
            'p95': float(np.percentile(arr, 95)),
            'p99': float(np.percentile(arr, 99)),
            'mean': float(np.mean(arr)),
            'max': float(np.max(arr)),
        }

    def compare_baseline(self, baseline_predictions: np.ndarray,
                         y_true: np.ndarray, task: str) -> dict:
        if task == 'regression':
            baseline_mae = float(np.mean(np.abs(baseline_predictions - y_true)))
            # Last-value baseline: predict the last known value (constant)
            last_value = np.full_like(y_true, y_true[0] if len(y_true) > 0 else 0.0)
            lv_mae = float(np.mean(np.abs(last_value - y_true)))
            # Mean baseline
            mean_val = np.full_like(y_true, y_true.mean())
            mean_mae = float(np.mean(np.abs(mean_val - y_true)))
            return {
                'model_mae': baseline_mae,
                'last_value_mae': lv_mae,
                'mean_baseline_mae': mean_mae,
                'improvement_vs_last_value': (lv_mae - baseline_mae) / (lv_mae + 1e-8),
                'improvement_vs_mean': (mean_mae - baseline_mae) / (mean_mae + 1e-8),
            }
        else:
            baseline_acc = float(np.mean(baseline_predictions == y_true))
            majority = int(np.bincount(y_true.astype(int)).argmax())
            majority_pred = np.full_like(y_true, majority)
            majority_acc = float(np.mean(majority_pred == y_true))
            return {
                'model_accuracy': baseline_acc,
                'majority_baseline_accuracy': majority_acc,
                'improvement': baseline_acc - majority_acc,
            }

    def _compute_regression_metrics(self, y_pred: np.ndarray, y_true: np.ndarray) -> dict:
        y_pred = np.asarray(y_pred, dtype=np.float64)
        y_true = np.asarray(y_true, dtype=np.float64)
        err = y_pred - y_true
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        # MAPE: avoid division by zero
        denom = np.abs(y_true)
        denom = np.where(denom < 1e-8, 1e-8, denom)
        mape = float(np.mean(np.abs(err) / denom) * 100.0)
        ss_res = np.sum(err ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2) + 1e-10
        r2 = float(1.0 - ss_res / ss_tot)
        return {'mae': mae, 'rmse': rmse, 'mape': mape, 'r2': r2}

    def _compute_classification_metrics(self, y_pred: np.ndarray, y_true: np.ndarray) -> dict:
        y_pred = np.asarray(y_pred).flatten().astype(int)
        y_true = np.asarray(y_true).flatten().astype(int)
        accuracy = float(np.mean(y_pred == y_true))

        classes = np.unique(y_true)
        if len(classes) <= 1:
            return {'accuracy': accuracy, 'f1': accuracy, 'precision': accuracy, 'recall': accuracy}

        f1_scores = []
        precisions = []
        recalls = []
        for c in classes:
            tp = float(np.sum((y_pred == c) & (y_true == c)))
            fp = float(np.sum((y_pred == c) & (y_true != c)))
            fn = float(np.sum((y_pred != c) & (y_true == c)))
            prec = tp / (tp + fp + 1e-10)
            rec = tp / (tp + fn + 1e-10)
            f1 = 2.0 * prec * rec / (prec + rec + 1e-10)
            f1_scores.append(f1)
            precisions.append(prec)
            recalls.append(rec)

        return {
            'accuracy': accuracy,
            'f1': float(np.mean(f1_scores)),
            'precision': float(np.mean(precisions)),
            'recall': float(np.mean(recalls)),
        }

    def _conformal_coverage(self, y_pred: np.ndarray, y_true: np.ndarray,
                             alpha: float = 0.05) -> float:
        # Split-conformal: compute residuals on first half, coverage on second
        n = len(y_pred)
        if n < 4:
            return 0.0
        mid = n // 2
        cal_residuals = np.abs(y_pred[:mid] - y_true[:mid]).max(axis=-1)
        q_level = np.ceil((1 - alpha) * (mid + 1)) / mid
        q_level = min(q_level, 1.0)
        q_hat = float(np.quantile(cal_residuals, q_level))
        test_residuals = np.abs(y_pred[mid:] - y_true[mid:]).max(axis=-1)
        coverage = float(np.mean(test_residuals <= q_hat))
        return coverage

    def _estimate_flops(self) -> int:
        # Rough FLOPs estimate: 2 * n_params per forward step (multiply-add)
        n_params = self.model.n_base_params()
        return 2 * n_params

    def print_summary(self):
        if not self.results:
            print("No benchmark results available.")
            return
        header = f"{'Dataset':<30} {'Task':<12} {'MAE':>10} {'RMSE':>10} {'MAPE%':>8} {'Acc':>8} {'P50ms':>8} {'P99ms':>8}"
        print("=" * len(header))
        print("VULGARIS BENCHMARK SUMMARY")
        print("=" * len(header))
        print(header)
        print("-" * len(header))
        for r in self.results:
            line = (f"{r.dataset_name:<30} {r.task:<12} "
                    f"{r.mae:>10.4f} {r.rmse:>10.4f} {r.mape:>8.2f} "
                    f"{r.accuracy:>8.4f} {r.latency_p50_ms:>8.2f} {r.latency_p99_ms:>8.2f}")
            print(line)
        print("=" * len(header))
        total_params = self.results[0].n_params if self.results else 0
        print(f"Total parameters: {total_params:,}")
        print(f"Memory (float64): {self.results[0].memory_mb:.1f} MB" if self.results else "")

    def to_json(self, path: str):
        import dataclasses
        data = [dataclasses.asdict(r) for r in self.results]
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        if self.verbose:
            print(f"Results saved to {path}")
