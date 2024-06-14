import os
import ray.tune.schedulers as schedulers
import torch
import torch.nn as nn
import torch.optim as optim 
from ray import train, tune, cluster_resources, init
from ray.tune import Trainable
from torch.utils.data import DataLoader

from ..data.handlertorch import TorchDataset
from ..utils.yaml_model_schema import YamlRayConfigLoader
from .predict import PredictWrapper

class TuneWrapper():
    def __init__(self,
                 config_path: str,
                 model_class: nn.Module,
                 data_path: str,
                 experiment_object: object,
                 max_gpus: int = None,
                 max_cpus: int = None,
                 max_object_store_mem: float = None,
                 max_mem: float = None,
                 ray_results_dir: str = None) -> None:
        """
        Initialize the TuneWrapper with the paths to the config, model, and data.
        """
        self.config = YamlRayConfigLoader(config_path).get_config()
        self.config["model"] = model_class
        self.config["experiment"] = experiment_object

        if not os.path.exists(data_path):
            raise ValueError("Data path does not exist. Given path:" + data_path)
        self.config["data_path"] = os.path.abspath(data_path)
        
        # build the tune config
        self.config["tune"]["tune_params"]["scheduler"] = getattr(schedulers, self.config["tune"]["scheduler"]["name"])( **self.config["tune"]["scheduler"]["params"])
        self.tune_config = tune.TuneConfig(**self.config["tune"]["tune_params"])

        # set ray cluster total resources (max) and per trial resuorces (single set/combination of hyperparameter) (parrallel actors maximum resources)
        self.max_gpus             = max_gpus
        self.max_cpus             = max_cpus
        self.max_object_store_mem = max_object_store_mem     # this is a special subset of the total usable memory that ray need for his internal work, by default is set to 30% of total memory usable
        self.max_mem              = max_mem
        #self.gpu_per_trial = self.config["tune"]["gpu_per_trial"]
        #self.cpu_per_trial = self.config["tune"]["cpu_per_trial"]

        # build the run config
        self.checkpoint_config = train.CheckpointConfig(checkpoint_at_end=True) #TODO implement checkpoiting
        self.run_config = train.RunConfig(checkpoint_config=self.checkpoint_config,
                                          storage_path=ray_results_dir
                                        )                                       #TODO implement run_config (in tune/run_params for the yaml file)
        
        self.tuner = self.tuner_initialization()

    def tuner_initialization(self) -> tune.Tuner:
        """
        Prepare the tuner with the configs.
        """

        # initialize the ray cluster with the limiter on CPUs, GPUs or memory if needed, otherwise everything that is available. None is what ray uses to get all resources available for either CPU, GPU or memory.
        # memory is split in two for ray. read more at ray.init documentation.
        init(num_cpus=self.max_cpus, num_gpus=self.max_gpus, object_store_memory=self.max_object_store_mem, _memory=self.max_mem)
        print("#####  CLUSTER resources ->  ", cluster_resources())

        """
        return tune.Tuner(tune.with_resources(TuneModel, resources={"cpu": self.cpu_per_trial, "gpu": self.gpu_per_trial}),
                            tune_config=self.tune_config,
                            param_space=self.config,
                            run_config=self.run_config,
                        )
        """
        return tune.Tuner(TuneModel,
                            tune_config= self.tune_config,
                            param_space=self.config,
                            run_config=self.run_config,
                        )


    def tune(self) -> None:
        """
        Run the tuning process.
        """

        return self.tuner.fit() 

class TuneModel(Trainable):

    def setup(self, config: dict) -> None:
        """
        Get the model, loss function(s), optimizer, train and test data from the config.
        """

        # Initialize model with the config params
        self.model = config["model"](**config["model_params"])

        # Add data path
        self.data_path = config["data_path"]

        # Use the already initialized experiment class      
        self.experiment = config["experiment"]

        # Get the loss function(s) from the config model params
        # Note that the loss function(s) are stored in a dictionary, 
        # where the keys are the key of loss_params in the yaml config file and the values are the loss functions associated to such keys.
        self.loss_dict = config["loss_params"]
        for key, loss_fn in self.loss_dict.items():
            try:
                self.loss_dict[key] = getattr(nn, loss_fn)()
            except AttributeError:
                raise ValueError(f"Invalid loss function: {loss_fn}, check PyTorch for documentation on available loss functions")
        

        # get the optimizer parameters
        optimizer_lr = config["optimizer_params"]["lr"]

        # get the optimizer from PyTorch
        self.optimizer = getattr(optim, config["optimizer_params"]["method"])(self.model.parameters(), lr=optimizer_lr)

        # get step size from the config
        self.step_size = config["tune"]['step_size']

        # get the train and validation data from the config
        # run dataloader on them
        self.batch_size = config['data_params']['batch_size']
        self.training = DataLoader(TorchDataset(self.data_path, self.experiment, split=0), batch_size=self.batch_size, shuffle=True)  # TODO need to check the reproducibility of this shuffling
        self.validation = DataLoader(TorchDataset(self.data_path, self.experiment, split=1), batch_size=self.batch_size, shuffle=True)

    def step(self) -> dict:
        """
        For each batch in the training data, calculate the loss and update the model parameters.
        This calculation is performed based on the model's batch function.
        At the end, return the objective metric(s) for the tuning process.
        """
        for step_size in range(self.step_size):
            for x, y, meta in self.training:
                # the loss dict could be unpacked with ** and the function declaration handle it differently like **kwargs. to be decided, personally find this more clean and understable.
                self.model.batch(x=x, y=y, optimizer=self.optimizer, **self.loss_dict)
        return self.objective()

    def objective(self) -> dict:
        """
        Compute the objective metric(s) for the tuning process.
        """
        metrics = ['loss', 'rocauc', 'prauc', 'mcc', 'f1score', 'precision', 'recall', 'spearmanr']  # TODO maybe we report only a subset of metrics, given certain criteria (eg. if classification or regression)
        predict_val = PredictWrapper(self.model, self.data_path, self.experiment,  split=1, batch_size=self.batch_size, loss_dict=self.loss_dict)
        predict_train = PredictWrapper(self.model, self.data_path, self.experiment, split=0, batch_size=self.batch_size, loss_dict=self.loss_dict)
        return {**{'val_'+metric : value for metric,value in predict_val.compute_metrics(metrics).items()},
                **{'train_'+metric : value for metric,value in predict_train.compute_metrics(metrics).items()}}

    def export_model(self, export_dir: str) -> None:
        torch.save(self.model.state_dict(), export_dir)

    def load_checkpoint(self, checkpoint_dir: str) -> None:
        self.model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "model.pt")))
        self.optimizer.load_state_dict(torch.load(os.path.join(checkpoint_dir, "optimizer.pt")))

    def save_checkpoint(self, checkpoint_dir: str) -> dict | None:
        torch.save(self.model.state_dict(), os.path.join(checkpoint_dir, "model.pt"))
        torch.save(self.optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
        return checkpoint_dir
    