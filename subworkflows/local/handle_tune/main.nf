/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    IMPORT NF-CORE MODULES/SUBWORKFLOWS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

include { TORCH_TUNE } from '../../../modules/local/torch_tune.nf'

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

    // add initial weights if provided
    if (input_initial_weights == null){
        model_conf_data = model_conf_data.map{ it -> [it[0], it[1], it[2], it[3], it[4], it[5], []] }
    }else{
        initial_weights = Channel.fromPath( input_initial_weights )
        model_conf_data = model_conf_data.combine(initial_weights)
    }

    // TUNE the torch model, TODO in future here switch TUNEing on basis of model type, keras tensorflow ecc.
    TORCH_TUNE( model_conf_data )

    // sort the debug out so that is in a deterministic order and can be used by nf-tests
    debug = null
    if ( params.debug_mode ) {
        debug = TORCH_TUNE.out.debug.toSortedList { a, b -> b[0] <=> a[0] }
    }

    emit:
    tune_out  = TORCH_TUNE.out.tune_specs
    model
    debug

}

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    THE END
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/
