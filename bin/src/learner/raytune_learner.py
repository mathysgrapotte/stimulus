import torch.nn as nn
import torch.optim as optim 
import torch
from torch.utils.data import DataLoader
from ray.tune import Trainable
from ..data.handlertorch import TorchDataset
from ray import train, tune
import ray.tune.schedulers as schedulers
import os
import json


        
class TuneTrainWrapper():
    def __init__(self, config_path: str, model_class: type, data_path: str, experiment_name: object) -> None:
        """
        Initialize the TuneWrapper with the paths to the config, model, and data.
        """
        # Load the config
        self.config = {}
        with open(config_path, "r") as f:
            # TODO figure out a better way to load the config
            self.config = json.load(f)
        
        self.config["model"] = model_class
        self.config["experiment"] = experiment_name

        try :
            assert os.path.exists(data_path)
            # make the path absolute so that ray does not complain on relative file paths
            self.config["data_path"] = os.path.abspath(data_path)
        except AssertionError:

            raise ValueError("Data path does not exist. Given path:" + data_path)           

        try:
            self.scheduler = getattr(schedulers, self.config["scheduler"]["name"])( **self.config["scheduler"]["params"])
        except AttributeError:
            raise ValueError(f"Invalid optimizer: {self.config['scheduler']['name']}, check PyTorch for documentation on available optimizers")

        # Get the tune config
        try: 
            self.tune_config = getattr(tune, self.config["tune_config"]["name"])(scheduler = self.scheduler, **self.config["tune_config"]["params"])
        except AttributeError:
            raise ValueError(f"Invalid tune_config: {self.config['tune_config']['name']}, check PyTorch for documentation on how a tune_config should be defined")

        # Get checkpoint config 
        try:
            # TODO: once the file reading is fixed, add the checkpoint_at_end parameter in the config file directly
            self.checkpoint_config = getattr(train, self.config["checkpoint_config"]["name"])(checkpoint_at_end = False, **self.config["checkpoint_config"]["params"])
        except AttributeError:
            raise ValueError(f"Invalid checkpoint_config: {self.config['checkpoint_config']['name']}, check PyTorch for documentation on how a checkpoint_config should be defined")

        # Get the run config
        try:
            self.run_config = getattr(train, self.config["run_config"]["name"])(checkpoint_config = self.checkpoint_config, **self.config["run_config"]["params"])
        except AttributeError:
            raise ValueError(f"Invalid run_config: {self.config['run_config']['name']}, check PyTorch for documentation on how a run_config should be defined")

        # substite in the checkpoint_config everything that is a bool in string with the actual bool
        self.tuner = None
        self.best_config = None
        self.results = None
        self.trainer = None

    def _prep_tuner(self) -> None:
        """
        Prepare the tuner with the configs.
        """
        self.tuner = tune.Tuner(TuneModel,
                            tune_config= self.tune_config,
                            param_space=self.config,
                            run_config=self.run_config,
                        )
        # set tuner to tuning mode
        # validation dataset is used for tuning
        self.tuner.tuning = True

    def tune(self, overwrite: bool = False) -> None:
        """
        Run the tuning process.
        """
        # This is just in case you want to tune more than once on the same initialization. Otherwise it will complain.
        if overwrite:
            self.tuner = None
            self.best_config = None
            
        if self.tuner is None:
            self._prep_tuner()
            self.results = self.tuner.fit()
            best_config = os.path.join(self.results.get_best_result().path, "params.json")
            try: 
                assert os.path.exists(best_config)
            except AssertionError:
                raise ValueError("Best config file not found.")
            with open(best_config, "r") as f:
                # TODO find better way to load the file
                self.best_config = json.load(f)
        else:
            raise ValueError("Tuner already exists - if you want to overwrite it, please set overwrite=True.")     

    def store_best_config(self, path: str) -> None:
        """
        Store the best config in a file.
        """
        with open(path, "w") as f:
            f.write(str(self.best_config))            
    
    def train(self) -> None: 
        """
        Train the model with the config.
        """
        
        config = None
        # use the config from the tuning if tune was run. Otherwise use the one fromn the initialization of the class.
        if self.best_config:
            config = self.best_config
            # reading the best_config file given by ray tune the model class is interpreted as a string, so it need to be resetted as class
            config["model"] = self.config["model"]
            # The same happens for the experiment object
            config["experiment"] = self.config["experiment"]
        else:
            config = self.config

        self.trainer = TuneModel(config=config)
        # set trainer to training mode (training dataset is used for training)
        self.trainer.tuning = False
        for i in range(config["epochs"]):
            self.trainer.step()


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
        # where the key is the name of the loss function and the value is the loss function itself.
        self.loss_dict = config["loss_fn"]
        for key, loss_fn in self.loss_dict.items():
            try:
                #self.loss_dict[key] = getattr(nn, loss_fn)()
                self.loss_dict[key] = loss_fn
            except AttributeError:
                raise ValueError(f"Invalid loss function: {loss_fn}, check PyTorch for documentation on available loss functions")
        
        # get the optimizer from the config
        try:
            self.optimizer = getattr(optim, config["optimizer"]["name"])(self.model.parameters(), **config["optimizer"]["params"])
        except AttributeError:
            raise ValueError(f"Invalid optimizer: {config['optimizer']['name']}, check PyTorch for documentation on available optimizers")

        # get epochs from the config
        self.epochs = config['epochs']

        # get step size from the config
        self.step_size = config['step_size']

        # get learning rate
        self.lr = config['lr']

        # get the train and validation data from the config
        # run dataloader on them
        self.training = DataLoader(TorchDataset(self.data_path, self.experiment, split=0), batch_size=config['data_params']['batch_size'])
        self.validation = DataLoader(TorchDataset(self.data_path, self.experiment, split=1), batch_size=config['data_params']['batch_size'])

        # The tuning variable is used to determine whether the model is being tuned or trained.
        # This is used to determin which data to use (training or validation).
        # TODO we could implement this outside of this class
        self.tuning = False

    def step(self) -> dict:
        """
        Train the model for one epoch.
        For each batch in the training data, calculate the loss and update the model parameters.
        This calculation is performed based on the model's step function.
        At the end, return the objective metric(s) for the tuning process.
        """
        
        if self.tuning: 
            dataloader = self.validation
        else:
            dataloader = self.training
        
        loss = 0.0
        for step_size in range(self.step_size):
            for x, y, meta in dataloader:
                self.optimizer.zero_grad()
                current_loss = self.model.step(x, y, self.loss_dict)
                loss += current_loss.item()
                current_loss.backward()
                self.optimizer.step()
            loss /= len(dataloader)
        return self.objective()

    def objective(self) -> dict:
        """
        Compute the objective metric(s) for the tuning process.
        """
        return {"val_loss": self.compute_val_loss()}

    def compute_val_loss(self) -> float:
        """
        Compute loss on the validation data.
        For each batch in the validation data, calculate the loss.
        This calculation is performed based on the model's step function.
        Then retun the average loss.
        """
        loss = 0.0
        self.model.eval()
        with torch.no_grad():
            for x, y, meta in self.validation:
                loss += self.model.step(x, y, self.loss_dict).item()
        loss /= len(self.validation)
        return loss
        
    def export_model(self, export_dir: str) -> None:
        torch.save(self.model.state_dict(), export_dir)

    def load_checkpoint(self, checkpoint: dict | None) -> None:
        if checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def save_checkpoint(self, checkpoint_dir: str) -> dict | None:
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        torch.save(checkpoint, checkpoint_dir)
        return checkpoint
        



        


