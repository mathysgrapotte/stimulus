
from abc import ABC, abstractmethod
from typing import Literal
from itertools import product

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


    def _transform_noise_dict(self):
        """
        TODO helper fucntion section
        """
        noise_dict = {}
        for col_name_dictionary in self.noise_arg:
            # The name: field of a noise: can be either a simlpe string or list of strings, so convert such variable to a list if it's a string, otherwise leave it unchanged
            noiser_list = [col_name_dictionary['name']] if isinstance(col_name_dictionary['name'], str) else col_name_dictionary['name']
            # Now get the parametrs or set of parameters associated with each noiser and store bot in a tuple and append to list noiser names associated to a given clumn_name
            for k, noiser_name in enumerate(noiser_list):
                # handle the fact that params can have "default" as value and not a list
                if col_name_dictionary['params'] == "default":
                    params_to_be_added = "default"
                else:
                    params_to_be_added =  col_name_dictionary['params'][k]
                # handle the case of multiple noiser with same name in the same list associated to the column_name, solution -> create a scheme to modify the name
                if noise_dict.get(col_name_dictionary["column_name"]) and noiser_name in noise_dict.get(col_name_dictionary["column_name"]) :
                    # Modify the noiser name already present appending a unique key to it
                    noiser_name = noiser_name + '-#' + str(k)
                #noise_dict.setdefault(col_name_dictionary["column_name"], []).append( {noiser_name : params_to_be_added} )
                noise_dict.setdefault(col_name_dictionary["column_name"], {})[noiser_name] = params_to_be_added
        return noise_dict


    def _generate_cartesian_product_combinations(self, d: dict) -> list:
        """
        Helper function for creating cartesian product combinations out of a dictionary.
        TODO expand explanation
        """
        keys = d.keys()
        value_lists = d.values()

        # Generate Cartesian product of value lists
        combinations = product(*value_lists)

        # Create dictionaries for each combination
        result = []
        for combination in combinations:
            combined_dict = {}
            for key, value in zip(keys, combination):
                nested_dict = {value : d[key][value]}
                combined_dict.update({key: nested_dict})
            result.append(combined_dict)

        return result


    
    def _handle_parameter_selection(self, d: dict, param_index: int) -> dict:
        """
        TODO helper fucntion section
        """
        
        for key, param_dict in d.items():
            # remove the appendix used to handle same noise names for same column_name, this is done in the _transform_noise_dict function, this line does nothing if that key is not present afterall
            key = key.split('-#')[0]
            # handle "defualt" as params value
            if param_dict == 'default':
                return {"name" : key, "params" : param_dict}
            else:
                tmp_param_dict = {}
                # iterate through the possible multiple parameter otpions
                for param_name, param_value in param_dict.items():
                    tmp_param_dict[param_name] = param_value[param_index]
                return {"name": key, "params": tmp_param_dict}  
                


    def noise_column_wise_combination(self) -> list:
        """
        works on the self.noise_arg dictionary to compute all column wise combinations for parametrs and noise function specified.
        The combinations of noisers is all against all, except there can not be two noisers for the same column_name.
        Combinations of noisers will always include at least one noiser per column_name.
        example for noisers ->

        column_name : 1                                  column_name : 2
        name : [noiser1, noiser2]                        name: [othernoiser]
        
        combinations ->
            noiser1 - othernoiser
            noiser2 - othernoiser

        Now this is how noiser functions are selected but for each of the above combination there are as many as there are parameters.
        Again an example shows it better ->

        column_name : 1                                                 column_name : 2
        name : [noiser1, noiser2]                                       name: [othernoiser]
        parameters : [{p1 : [1 ,2 ,3]}, {p1 : [1.5, 2.5, 3.5 ]}]        parameters : [{p1 : [4 ,5 ,6], p2 : [7, 8, 9]}]

        combinations ->
            noiser1 (p1 = 1) - othernoiser (p1 = 4, p2 = 7)
            noiser1 (p1 = 2) - othernoiser (p1 = 5, p2 = 8)
            noiser1 (p1 = 3) - othernoiser (p1 = 6, p2 = 9)
            noiser2 (p1 = 1.5) - othernoiser (p1 = 4, p2 = 7)
            noiser2 (p1 = 2.5) - othernoiser (p1 = 5, p2 = 8)
            noiser2 (p1 = 3.5) - othernoiser (p1 = 6, p2 = 9)
        """

        # transform noise entry in a nested dictionary, with structure {col_name: { noiser_name : {parameters : {p1 : [1]} }}}
        noise_as_dict = self._transform_noise_dict()
            
        # Create cartesian product of noiser names based on the above dictionary
        noiser_combination_list = self._generate_cartesian_product_combinations(noise_as_dict)

        # for each noiser combination create the column wise selection of parameters associated
        all_noise_combination = []
        for noise_combo in noiser_combination_list:
            # select the parameter iterating through the total number of parameters value fopr each col type 
            for params_index in range(self.number_culumn_wise_val):
                noise_list = []
                for col_name, noise_dict in noise_combo.items():
                    single_param_dict = self._handle_parameter_selection(noise_dict, params_index)
                    # add the column_name field to this dictionary
                    single_param_dict["column_name"] = col_name
                    # reorder the entries by key alphabetically for readability
                    sorted_dict = {key: single_param_dict[key] for key in sorted(single_param_dict)}
                    noise_list.append(sorted_dict)
                all_noise_combination.append({'noise' : noise_list })
        return all_noise_combination



    def noise_all_combination(self) -> list:
        """
        works on the self.noise_arg dictionary to compute all possible combinations of parameters and nboisers in a all against all fashion.
        """

        # TODO implement this function
        raise ValueError("the function noise_all_combination for the flag interpret_parmas_mode : all_combinations is not implemented yet ")
            
    

