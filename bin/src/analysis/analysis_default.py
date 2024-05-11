import math
import pandas as pd
from matplotlib import pyplot as plt
from typing import Any, Tuple
from src.learner.predict import PredictWrapper

class AnalysisPerformanceTune:
    """
    Report the performance during tuning.
    """
    def __init__(self, results_path: str):
        self.results = pd.read_csv(results_path)

    def plot_metric_vs_iteration(self, metrics: list, figsize: tuple = (10,10), output: str = None):

        # create figure
        rows, cols = self.get_grid_shape(len(metrics))
        fig, axs = plt.subplots(rows, cols, figsize=figsize)

        # plot each metric
        for i,ax in enumerate(axs.flat):
            if i >= len(metrics):
                ax.axis('off')
                continue
            self.plot_metric_vs_iteration_per_metric(axs.flat[i], metrics[i])

        # add legend
        # axs.flat[0].legend()
        handles, labels = axs[0, 0].get_legend_handles_labels()  # Get handles and labels from one subplot
        plt.legend(handles, labels, loc='upper left')  # Adjust location as needed

        # save plot
        plt.tight_layout()
        if output:
            plt.savefig(output)
        plt.show()
    
    def plot_metric_vs_iteration_per_metric(self, ax: Any, metric: str):
        """
        Plot the metric vs the iteration.
        """

        # plot training performance
        ax.plot(
            self.results.training_iteration,
            self.results['train_'+metric],
            c='blue',
            label='train'
        )

        # plot validation performance
        ax.plot(
            self.results.training_iteration,
            self.results['val_'+metric],
            c='orange',
            label='val'
        )

        # TODO set x-axis labels into integer
        # plt.xticks(range(min(self.results.training_iteration), max(self.results.training_iteration)))

        # add labels
        ax.set_xlabel('epoch')
        ax.set_ylabel(metric)

        return ax
    
    @staticmethod
    def get_grid_shape(n: int) -> Tuple[int,int]:
        """Calculates rows and columns for a rectangle layout (flexible)."""
        rows = int(math.ceil(math.sqrt(n)))  # Round up the square root for rows
        cols = int(math.ceil(n / rows))  # Calculate columns based on rows
        return rows, cols

class AnalysisPerformanceModel:
    """
    Provide the performance of a model.
    """
    def __init__(self, model: object, data_path: str, experiment: object, batch_size=None):
        self.model = model
        self.data_path = data_path
        self.experiment = experiment
        self.predictor = PredictWrapper(self.model, self.data_path, self.experiment, split=2, batch_size=batch_size)
        
    def get_performance_table(self, metrics: list, output: str = None) -> Tuple[pd.DataFrame,None]:
        """
        Compute the performance metrics and create a table for it.
        """
        perf = {}
        for metric in metrics:
            perf[metric] = self.predictor.compute_metric(metric)
        perf = pd.DataFrame(perf, index=[0])
        if output:
            perf.to_csv(output, index=False)
        return pd.DataFrame(perf)

# class AnalysisRobustness:
#     def __init__(self, data):
#         pass