
from abc import ABC, abstractmethod
from typing import Literal
from itertools import product

class JsonSchema(ABC):
    """
    This class helps decode and work on a difened Json schema used by the stimulus pipeline.
    TODO add Json.schema real library to control that each noise, split have the correct keys associated to them.
    link -> https://json-schema.org/learn/getting-started-step-by-step#create
    """
    def __init__(self, schema: dict ) -> None:
        self.schema                = schema
        self.interpret_params_mode = schema.get('interpret_params_mode', 'column_wise')
        self.experiment            = schema.get('experiment', None)
        self.noise_arg             = schema.get('noise', [])
        self.split_arg             = schema.get('split', [])
        self.custom_arg            = schema.get('custom', [])

        # Send error if experiment name is missing
        if not self.experiment:
            raise ValueError(f"No experiment name given, the Json should always have a experiment:'ExperimentName' field")

        # Send error if self.interpret_parmas_mode is not among the possible ones
        if self.interpret_params_mode not in ["column_wise", "all_combinations"]:
            raise ValueError(f"interpret_params_mode value can only be one of the following keywords -> ['column_wise', 'all_combinations']")

        # check that inside noise dictionary there are no repeated column_nmae values and return them otherwise send error
        self.column_names = self._check_repeated_column_names()

        # check that noise dictionary have a coherent number of parameters values in case of column_wise for self.interpret_parmas_mode
        self.number_column_wise_val = self._check_noise_params_schema()



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



    def _check_noise_params_schema(self) -> int:
        """
        Help function to check if the number of values in params in the noise dictionary is consistent among all params.
        If there is {"NoiserName" : { "params": [{"val1":[0, 1]}], "OtherNoiser" : { "params": [{"val1":[2, 3], "val3":[4]}]}}
        it will raise error because the val3 has only a list of len() 1 instead of 2
        otherwise it resturn the len()
        """

        # in case there is no noise dictionary or if interpret_params_mode is in all_combinations mode
        if not self.noise_arg  or self.interpret_params_mode == 'all_combinations' :
            return 0

        num_params_list = []
        # Iterate through the given dictionary becuse more than one column_name values could be specified for ex.
        for i, col_name_dictionary in enumerate(self.noise_arg):
            
            # take into account that there could be the keyword default
            if col_name_dictionary["params"] == "default":
                continue

            # iterate throught the possible multiple parmaeters, some noisers could have more than one parameter flag
            else:
                for k, params_dict in enumerate(col_name_dictionary["params"]):
                    # even the single set of parameters of a given noisername can be set to default
                    if params_dict == "default":
                        continue
                    for params_flag, params_list in params_dict.items():
                        num_params_list.append(len(params_list))
        
        # check that all parameters values found are equal
        if len(set(num_params_list)) == 1:
            return num_params_list[0]
        else:
            raise ValueError(f"Expected the same number of values for all the params under noise value, but received a discordant ammount instead.")



    def _transform_noise_dict(self) -> dict:
        """
        TODO helper fucntion section
        """
        noise_dict = {}
        for col_name_dictionary in self.noise_arg:
            # The name: field of a noise: can be either a simlpe string or list of strings, so convert such variable to a list if it's a string, otherwise leave it unchanged
            noiser_list = [col_name_dictionary['name']] if isinstance(col_name_dictionary['name'], str) else col_name_dictionary['name']
            # Now get the parametrs or set of parameters associated with each noiser and store both in a tuple and append to list noiser names associated to a given clumn_name
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
            # flag to check if all the parameters values associated to one combination of nopiser are all default
            all_param_value_default = True
            for key, value in zip(keys, combination):
                param_field =  d[key][value] 
                nested_dict = {value : param_field}
                combined_dict.update({key: nested_dict})
                # now check if the param is a default or not 
                if param_field != "default":
                    all_param_value_default = False  
            # now append the value to the combo dict that rapresent how many parameters combination there are for such noisers combination.
            tmp_tuple = (combined_dict, self.number_column_wise_val)
            if all_param_value_default:
                tmp_tuple = (combined_dict, 1)       
            result.append(tmp_tuple)

        return result


    
    def _handle_parameter_selection(self, d: dict, param_index: int) -> dict:
        """
        TODO helper fucntion section
        """
        
        for key, param_dict in d.items():
            # remove the appendix used to handle same noise names for same column_name, this is done in the _transform_noise_dict function, this line does nothing if that key is not present afterall
            key = key.split('-#')[0]
            # handle "defualt" as params value returning a empty dict 
            if param_dict == 'default':
                return {"name" : key, "params" : {}}
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

        # transform noise entry in a nested dictionary, with structure {col_name: { noiser_name : {p1 : [1]} }} 
        noise_as_dict = self._transform_noise_dict()
            
        # Create cartesian product of noiser names based on the above dictionary and check if the single combination does not fall under the special case where all parametrs associated to each noisers in the combination are set to "default". in such a case the code that follows in a specific for loop should be executed only once, instead of self.number_column_wise_val times.
        noiser_combination_list = self._generate_cartesian_product_combinations(noise_as_dict)

        # for each noiser combination create the column wise selection of parameters associated
        all_noise_combination = []
        for noise_combo_tuple in noiser_combination_list:
            # select the parameter iterating through the total number of parameters associated to the specific noiser combination under selection. This value is the second value of the tuple in which the actual dictionary of noiser combination is.
            for params_index in range(noise_combo_tuple[1]):
                noise_list = []
                # noise_combo_tuple[0] is the actual dictionary with the noiser name parameters for the given combination
                for col_name, noise_dict in noise_combo_tuple[0].items():
                    single_param_dict = self._handle_parameter_selection(noise_dict, params_index)
                    # add the column_name field to this dictionary
                    single_param_dict["column_name"] = col_name
                    # reorder the entries by key alphabetically for readability
                    sorted_dict = {key: single_param_dict[key] for key in sorted(single_param_dict)}
                    noise_list.append(sorted_dict)
                all_noise_combination.append(noise_list)

        return all_noise_combination



    def noise_all_combination(self) -> list:
        """
        works on the self.noise_arg dictionary to compute all possible combinations of parameters and nboisers in a all against all fashion.
        """

        # TODO implement this function
        raise ValueError("the function noise_all_combination for the flag interpret_parmas_mode : all_combinations is not implemented yet ")
            
    

    def split_combination(self) -> list:
        """
        TODO add description
        """

        list_split_comibinations = []
        # iterate through the split entry and return a list of split possibilities, where each splitter_name has one/set of one parametyers
        for i, split_dict in enumerate(self.split_arg):
            # jsut create a new dictionary for each set of params associated to each split_name, basically if a splitter has more than one element in his params: then they should be decoupled so to have each splitter with only one value for params:
            # if the value of params: is "default" just return the dictionary  with an empty dict as value of params : 
            if split_dict['params'] == "default" or split_dict['params'] == ["default"]:
                split_dict['params'] = {}
                list_split_comibinations.append( split_dict )
            else:
                # Get lengths of all lists
                lengths = {key: len(value) for key, value in split_dict['params'][0].items()}

                # Check if all lengths are the same
                all_lengths_same = set(lengths.values())

                if len(all_lengths_same) != 1 :
                    raise ValueError(f"All split params for the same splitter have to have the same number of elements, this splitter does not: {split_dict['name']}.")
                else:
                    # iterate at level of number of params_values 
                    for params_index in range(list(all_lengths_same)[0]):
                        # making the split into a dict the _handle_parameter_selection can use
                        single_param_dict = self._handle_parameter_selection({split_dict['name']: split_dict['params'][0] }, params_index)
                        list_split_comibinations.append(single_param_dict)
        return list_split_comibinations