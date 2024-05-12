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

    main:
    // put the files in channels, 
    model       = Channel.fromPath( input_model  ).first()  // TODO implement a check and complain if more that one file is pased as model
    tune_config = Channel.fromPath( input_tune_config )

    // assign a model and a TUNE_config to each data
    model_conf_pair = model.combine(tune_config)
    model_conf_data = model_conf_pair.combine(data).map{ it -> [it[2], it[1], it[0], it[4], it[3]]}   // just reordering according to the inputs of the launch_TUNEing.py
    
    // TUNE the torch model, TODO in future here switch TUNEing on basis of model type, keras tensorflow ecc.
    TORCH_TUNE( model_conf_data )

    emit:
    model  = TORCH_TUNE.out.tune_specs
}


/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    THE END
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/