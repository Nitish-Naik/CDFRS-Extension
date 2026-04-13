import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# =============================================================================
# Academic Formatting Setup
# =============================================================================
def set_academic_style():
    """Configures matplotlib for academic conference papers (IEEE/ACM style)."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Computer Modern Roman", "serif"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "figure.dpi": 300,        # High res for papers
        "savefig.dpi": 300,
        "savefig.bbox": "tight",  # Removes excess whitespace
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--"
    })

# =============================================================================
# Plotting Suite
# =============================================================================
import os

class EnsemblePlotter:
    def __init__(self):
        # Relaxed font requirements so the Linux server doesn't complain
        plt.rcParams.update({
            "font.family": "serif", "font.size": 11, "axes.titlesize": 12,
            "axes.labelsize": 11, "legend.fontsize": 10, "lines.linewidth": 2,
            "figure.dpi": 300, "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--"
        })
        self.windows = []
        self.metrics = {"auc": [], "f1": [], "accuracy": []}
        self.static_metrics = {"accuracy": []}
        self.swd_scores = []
        self.drift_flags = []
        self.weights = {"dt": [], "rf": [], "gbt": [], "lr": []}
        self.null_distributions = {}
        self.final_y_true = []
        self.final_y_pred = []
        
        # Create a directory to save the plots
        os.makedirs("academic_plots", exist_ok=True)

    def generate_all_plots(self):
        # 1. Metrics over time
        plt.figure(figsize=(7, 4))
        plt.plot(self.windows, self.metrics["auc"], marker='o', label='AUC', color='#1f77b4')
        plt.plot(self.windows, self.metrics["f1"], marker='s', linestyle='--', label='F1', color='#ff7f0e')
        plt.plot(self.windows, self.metrics["accuracy"], marker='^', linestyle='-.', label='Accuracy', color='#2ca02c')
        plt.xlabel('Data Window (Batch Index)'); plt.ylabel('Score'); plt.title('Ensemble Performance per Window')
        plt.legend(loc='lower right'); plt.ylim(0.0, 1.05)
        plt.savefig("academic_plots/1_metrics_over_time.pdf", bbox_inches='tight')
        plt.close()

        # 2. SWD and Drift
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(self.windows, self.swd_scores, marker='o', color='black', label='Observed SWD')
        drift_x = [w for w, flag in zip(self.windows, self.drift_flags) if flag]
        drift_y = [s for s, flag in zip(self.swd_scores, self.drift_flags) if flag]
        ax.scatter(drift_x, drift_y, color='red', s=100, zorder=5, label='Drift Detected', marker='X')
        for x in drift_x: ax.axvline(x=x, color='red', linestyle=':', alpha=0.5)
        ax.set_xlabel('Batch Index'); ax.set_ylabel('SWD'); ax.set_title('Concept Drift Evolution'); ax.legend()
        plt.savefig("academic_plots/2_swd_and_drift.pdf", bbox_inches='tight')
        plt.close()

        # 3. Static vs Dynamic Accuracy
        if len(self.static_metrics["accuracy"]) == len(self.metrics["accuracy"]):
            plt.figure(figsize=(7, 4))
            plt.plot(self.windows, self.metrics["accuracy"], marker='o', label='Dynamic Ensemble', color='#d62728')
            plt.plot(self.windows, self.static_metrics["accuracy"], marker='x', linestyle='--', label='Static Baseline', color='#7f7f7f')
            plt.xlabel('Batch Index'); plt.ylabel('Accuracy'); plt.title('Dynamic vs Static Model Performance'); plt.legend()
            plt.savefig("academic_plots/3_static_vs_dynamic.pdf", bbox_inches='tight')
            plt.close()

        # 4. Ensemble Weights Composition
        plt.figure(figsize=(8, 4.5))
        y_stacked = np.vstack([self.weights[k] for k in self.weights.keys()])
        plt.stackplot(self.windows, y_stacked, labels=[k.upper() for k in self.weights.keys()], alpha=0.8)
        plt.xlabel('Batch Index'); plt.ylabel('Ensemble Weight'); plt.title('Ensemble Composition Over Time')
        plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1)); plt.margins(x=0, y=0)
        plt.savefig("academic_plots/4_ensemble_weights.pdf", bbox_inches='tight')
        plt.close()

        # 5. Permutation Test (for first drift detected)
        if drift_x and drift_x[0] in self.null_distributions:
            null_dist, obs_swd = self.null_distributions[drift_x[0]]
            plt.figure(figsize=(6, 4))
            sns.histplot(null_dist, kde=True, color='gray', stat='density', alpha=0.5, label='Null Distribution')
            plt.axvline(obs_swd, color='red', linestyle='dashed', linewidth=2, label=f'Observed SWD ({obs_swd:.4f})')
            plt.xlabel('SWD'); plt.ylabel('Density'); plt.title(f'Permutation Test (Window {drift_x[0]})'); plt.legend()
            plt.savefig("academic_plots/5_permutation_test.pdf", bbox_inches='tight')
            plt.close()

        # 6. Confusion Matrix
        if self.final_y_true:
            cm = confusion_matrix(self.final_y_true, self.final_y_pred)
            plt.figure(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Benign', 'Attack'], yticklabels=['Benign', 'Attack'])
            plt.xlabel('Predicted Label'); plt.ylabel('True Label'); plt.title('Aggregate Confusion Matrix')
            plt.savefig("academic_plots/6_confusion_matrix.pdf", bbox_inches='tight')
            plt.close()
            
        print("[Done] All plots saved to the 'academic_plots' directory as PDFs.")