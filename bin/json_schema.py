
from abc import ABC, abstractmethod
from typing import Literal 

class JsonSchema(ABC):
    """
    This class helps decode and work on a difened Json schema used by the stimulus pipeline
    """
    def __init__(self, schema: dict ) -> None:
        self.schema = schema
        self.noise_arg  = schema.get('noise', [])
        self.split_arg  = schema.get('split', [])
        self.custom_arg = schema.get('custom', [])

        # check that both noise and split have they're coherent number of parameters values
        self.number_noise_val = self._check_params_schema('noise')
        self.number_split_val = self._check_params_schema('split')


    def _check_params_schema(self, switch: Literal['noise', 'split']) -> int:
        """
        Help function to check if the number of values in params in the noise dictionary is consisten among all params.
        If there is {"Noisernmae" : { "params": [{"val1":[0, 1]}], "OtherNoiser" : { "params": [{"val1":[2, 3], "val3":[4]}]}}
        it will raise error because the val3 has only a list of len() 1 instead of 2
        otherwise it resturn the len()
        """

        starting_list = self.noise_arg
        if switch == 'split':
            starting_list = self.split_arg

        # in case there is no noise or split flag but a custom one instead
        if not starting_list and self.custom_arg:
            return None

        num_params_list = []
        # Iterate through the given dictionary becuse more than one noising function could be specified for ex.
        for i, dictionary in enumerate(starting_list):
            
            # take into account that there could be the keyword default
            if dictionary["params"] == "default":
                # TODO think what to do in this case
                continue

            # iterate throught the possible multiple parmaeters
            else:
                for params_flag, params_list in dictionary["params"][0].items():
                    num_params_list.append(len(params_list))

        # check that all parameters values found are equal
        if len(set(num_params_list)) == 1:
            return num_params_list[0]
        else:
            raise ValueError(f"Expected the same number of values for all the params under {switch} flag, but received a discordant ammount input Json.")

            
    

