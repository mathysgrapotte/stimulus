/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    IMPORT NF-CORE MODULES/SUBWORKFLOWS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

include { CHECK_TORCH_MODEL } from '../modules/local/check_torch_model.nf'

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    RUN MAIN SUBWORKFLOW
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

workflow CHECK_MODEL {

    take:
    input_csv
    input_json
    input_model
    input_tune_config
    
    main:

    completion_message = "\n###\nThe check of the model has been skipped.\n###\n"

    // It can be still skipped, but by default is run. 
    if ( params.check_model ) {
        
        // put the files in channels
        csv         = Channel.fromPath( input_csv  )
        model       = Channel.fromPath( input_model  ).first()  // TODO implement a check and complain if more that one file is pased as model
        json        = Channel.fromPath( input_json ) 
        tune_config = Channel.fromPath( input_tune_config )
        

        // combine everything toghether in a all vs all manner
        model_tuple = csv.combine(model)
            .combine(json)
            .combine(tune_config)
        
        // launch the check using torch. TODO put selection of module based on type: torc, tensorflow ecc..
        CHECK_TORCH_MODEL( model_tuple ) 
        completion_message = CHECK_TORCH_MODEL.out.standardout
        
    }

    // to enforce that the second run workflow depends on this one
    emit:
    completion_message

}


/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    THE END
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/
