/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    IMPORT NF-CORE MODULES/SUBWORKFLOWS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

include { TORCH_TUNE } from '../modules/local/torch_tune.nf'


/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    RUN MAIN WORKFLOW
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

workflow HANDLE_TUNE {

    take:
    input_model
    input_tune_config
    data
    input_initial_weights

    main:
    // put the files in channels, 
    model       = Channel.fromPath( input_model  ).first()  // TODO implement a check and complain if more that one file is pased as model
    tune_config = Channel.fromPath( input_tune_config )

    // assign a model and a TUNE_config to each data
    model_conf_pair = model.combine(tune_config)
    model_conf_data = model_conf_pair.combine(data).map{ 
        it -> [it[2], it[3], it[1], it[0], it[5], it[4]]
    }  // just reordering according to the inputs of the launch_tuning.py

    // add initial weights if provided and drop split_key as not needed by the process, but needed later for analysis.
    if (input_initial_weights == null){
        // the empty list will be passed to path operator which will return "" and simply not pass anything to the louncher
        model_conf_data_weights = model_conf_data.map{ it -> [it[0] + " - no_innitial_weights", it[0], it[2], it[3], it[4], it[5], []] }
    }else{
        initial_weights = Channel.fromPath( input_initial_weights )
        // differently from above now we have to combine the two channels and add the initial weights filename to the combination_key. This is usefull for output file renaming.
        model_conf_data_weights = model_conf_data.combine(initial_weights).map{
            it -> [it[0] + " - " + it[6].getName(), it[0], it[2], it[3], it[4], it[5], it[6]]
            }
    }

    // TUNE the torch model, TODO in future here switch TUNEing on basis of model type, keras tensorflow ecc.
    TORCH_TUNE( model_conf_data_weights )

    // prepare the data for STYMULUS_ANALYSIS_DEFAULT. this means having a channel with as many elements as there are types of split. 
    // Each element will be a tuple with in order : split_key, list of combination_key, list of datafiles, list of experiment_config, list of tune_selected_hyperparam_conf, list of model_weights.pt, list of optimizer, list of metrics
    // if params.initial_weights contained more than one file then sublists will have either two lengths. All the lists of tune.out specific files will have length num_datafiles * num_initial_weightsfiles. All other lists will have length num_datafiles.
    tune_out_grouped      = TORCH_TUNE.out.tune_specs.groupTuple()
    tune_out_for_analysis = model_conf_data.join( tune_out_grouped )
        .groupTuple(by: 1)
        .map{ 
        it -> [it[1], it[6].flatten(), it[4], it[5], it[7].flatten(), it[8].flatten(), it[9].flatten(), it[10].flatten()]
        }

    // prepare the output of tune to have as many task of tune with each being a list of in order : combination_key, split_key, model.py, data (transformed), experiment_config, tune_selected_hyperparam_conf, model_weights.pt, optimizer, metrics
    tune_out_plain        = model_conf_data.combine( TORCH_TUNE.out.tune_specs, by: 0 )
        .map{
        it -> [it[6], it[1], it[3], it[4], it[5], it[7], it[8], it[9], it[10]]
    }

    // sort the debug out so that is in a deterministic order and can be used by nf-tests
    debug = null
    if ( params.debug_mode ) {
        debug = TORCH_TUNE.out.debug.toSortedList { a, b -> b[0] <=> a[0] }
    }

    emit:
    tune_out_for_analysis
    tune_out_plain
    model
    debug
    
}


/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    THE END
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/