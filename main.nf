#!/usr/bin/env nextflow
/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    model-check     help section
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

nextflow.enable.dsl = 2


// this prints the help in case you use --help parameter in the command line and it stops the pipeline
if (params.help) {
    log.info "\nThis is the help section of the pipeline, accessed using --help flag from command line."
    log.info "Here follows a description of the functionality and all the flags of the pipeline.\n"
    log.info ""
    exit 1
}

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    input handling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/



/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    NAMED WORKFLOW FOR PIPELINE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

include { CHECK_MODEL } from './subworkflows/check_model.nf'
include { HANDLE_DATA } from './workflows/handle_data.nf'
include { HANDLE_TUNE } from './workflows/handle_tune.nf'
include { HANDLE_ANALYSIS } from './workflows/handle_analysis.nf'

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    RUN ALL WORKFLOWS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/


workflow {

    CHECK_MODEL (
        params.csv,
        params.exp_conf,
        params.model,
        params.tune_conf,
        params.initial_weights
    )
    completion_message = CHECK_MODEL.out.completion_message

    HANDLE_DATA( 
        params.csv,
        params.exp_conf,
        completion_message
    )
    prepared_data = HANDLE_DATA.out.data
    
    HANDLE_TUNE(
        params.model,
        params.tune_conf,
        prepared_data,
        params.initial_weights
    )

    // this part works, but the docker container is not updated with matplotlib yet
    HANDLE_ANALYSIS(
        HANDLE_TUNE.out.tune_out_for_analysis,
        HANDLE_TUNE.out.model,
        params.initial_weights
    ) 
    
}


workflow.onComplete {
    println "Done"
}

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    THE END
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

