/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    IMPORT NF-CORE MODULES/SUBWORKFLOWS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

include { CHECK_TORCH_MODEL } from '../modules/local/stimulus_shuffle_csv.nf'

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
    
    completition_message = "\n###\nThe check of the model has been skipped.\n###\n"

    // It can be still skipped, but by default is run. 
    if ( params.check_model ) {
        
        // put the files in channels
        json        = Channel.fromPath( input_json )
        csv         = Channel.fromPath( input_csv  )
        model       = Channel.fromPath( input_model  ).first()  // TODO implement a check and complain if more that one file is pased as model
        tune_config = Channel.fromPath( input_tune_config )

        // read the json and extract the experiment name. (only real information needed from it). This lines below could be improved by finding a way to better handle ImmutableCollections$MapN class in java.
        experiment_name = json.splitJson().map { 
                it -> ["${it.values()[0]}", "${it.values()[1]}"] 
            }.filter{ 
                it[0] == "experiment"
            }.map{
                it -> it[1]
            }

        // combine everything toghether in a all vs all manner
        tmp1        = experiment_name.combine(csv)
        tmp2        = tmp1.combine(model)
        model_tuple = tmp2.combine(tune_config)
        
        // launch the check using torch. TODO put selection of module based on type: torc, tensorflow ecc..
        CHECK_TORCH_MODEL( model_tuple ) 

    }

    // to enforce dependency oh the rest of the workflows in the pipeline
    //emit:
    

}


/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    THE END
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/
