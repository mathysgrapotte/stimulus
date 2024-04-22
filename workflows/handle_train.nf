/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    IMPORT NF-CORE MODULES/SUBWORKFLOWS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

include { TORCH_TRAIN } from '../modules/torch_train.nf'

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    RUN MAIN WORKFLOW
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

workflow HANDLE_TRAIN {

    take:

    input_model
    input_train_config
    data


    main:

    // put the files in channels, 
    model        = Channel.fromPath( input_model  ).first()  // TODO implement a check and complain if more that one file is pased as model
    train_config = Channel.fromPath( input_train_config )

    // assign a madel and a train_config to each data
    model_conf_pair = model.combine(train_config)
    model_conf_data = model_conf_pair.combine(data).map{ it -> [it[2], it[1], it[0], it[4], it[3]]}   // just reordering according to the inputs of the launch_training.py
    
    // train the torch model, TODO in future here switch training on basis of model type, keras tensorflow ecc.
    TORCH_TRAIN( model_conf_data )

    emit:

    debug = TORCH_TRAIN.out.standardout
    //data  = train_config

}


/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    THE END
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/