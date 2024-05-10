import pandas as pd
from matplotlib import pyplot as plt
from typing import Tuple
from src.learner.predict import PredictWrapper

class AnalysisPerformanceTune:
    """
    Report the performance during tuning.
    """
    def __init__(self, results_path: str):
        self.results = pd.read_csv(results_path)
    
    def plot_metric_vs_iteration(self, metric: str, output: str = None) -> None:
        """
        Plot the metric vs the iteration.
        """
        # set up figure
        fig, ax = plt.subplots()

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
        ax.set_ylabel('loss')

        # add legend
        ax.legend()

        # save figure
        plt.show()
        if output:
            plt.savefig(output)
            plt.close()
        

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