
from abc import ABC, abstractmethod
from typing import Literal 

class JsonSchema(ABC):
    """
    This class helps decode and work on a difened Json schema used by the stimulus pipeline
    """
    def __init__(self, schema: dict ) -> None:
        self.schema                = schema
        self.interpret_params_mode = schema.get('interpret_parmas_mode', "culumn_wise")
        self.experiment            = schema.get('experiment', None)
        self.noise_arg             = schema.get('noise', [])
        self.split_arg             = schema.get('split', [])
        self.custom_arg            = schema.get('custom', [])

        # Send error if experiment name is missing
        if not self.experiment:
            raise ValueError(f"No experiment name given, the Json should always have a experiment:'ExperimentName' field")

        # Send error if self.interpret_parmas_mode is not of possibility
        if self.interpret_params_mode not in ["culumn_wise", "all_combinations"]:
            raise ValueError(f"interpret_params_mode value can only be one of the following keywords -> ['culumn_wise', 'all_combinations']")

        # check that inside noise dictionary there are no repeated column_nmae values and return them otherwise send error
        self.column_names = self._check_repeated_column_names()


        # check that noise dictionary have a coherent number of parameters values in case of column_wise for self.interpret_parmas_mode
        self.number_culumn_wise_val = self._check_params_schema()
  
    def _check_repeated_column_names(self) -> list:
        """
        Helper function that ensures that inside noise dictionary there are no column:names repeated values
        """

        # in case there is no noise or split flag but a custom one instead
        if not self.noise_arg and self.custom_arg:
            return None

        column_name_list = []
        for i, dictionary in enumerate(self.noise_arg):
            column_name = dictionary["column_name"]
            
            # If already present as a name throw an error
            if column_name in column_name_list:
                raise ValueError(f"The column_name {column_name} is repeated. column_names should be unique.")
            else:
                column_name_list.append(column_name)
        return column_name_list



    def _check_params_schema(self) -> int:
        """
        Help function to check if the number of values in params in the noise dictionary is consisten among all params.
        If there is {"NoiserName" : { "params": [{"val1":[0, 1]}], "OtherNoiser" : { "params": [{"val1":[2, 3], "val3":[4]}]}}
        it will raise error because the val3 has only a list of len() 1 instead of 2
        otherwise it resturn the len()
        """

        # in case there is no noise dictionary but a custom one instead or if interpret_params_mode is in all_combinations mode
        if (not self.noise_arg and self.custom_arg) or self.interpret_params_mode == 'all_combinations' :
            return None

        num_params_list = []
        # Iterate through the given dictionary becuse more than one column_name values could be specified for ex.
        for i, col_name_dictionary in enumerate(self.noise_arg):
            
            # take into account that there could be the keyword default
            if col_name_dictionary["params"] == "default":
                # TODO think what to do in this case
                continue

            # iterate throught the possible multiple parmaeters, some noisers could have more than one parameter flag
            else:
                for k, params_dict in enumerate(col_name_dictionary["params"]):
                    for params_flag, params_list in params_dict.items():
                        num_params_list.append(len(params_list))
        
        # check that all parameters values found are equal
        if len(set(num_params_list)) == 1:
            return num_params_list[0]
        else:
            raise ValueError(f"Expected the same number of values for all the params under noise value, but received a discordant ammount instead.")
        

    def noise_column_wise_combination(self) -> list:
        """
        works on the self.noise_arg dictionary to compute all column wise combinations for parametrs and noise function specified.
        """

        print(self.noise_arg)
        return []


    def noise_all_combination(self) -> list:
        """
        works on the self.noise_arg dictionary to compute all possible combinations of parameters and nboisers in a all against all fashion.
        """

        # TODO implement this function
        raise ValueError("the function noise_all_combination for the flag interpret_parmas_mode : all_combinations is not implemented yet ")
            
    

