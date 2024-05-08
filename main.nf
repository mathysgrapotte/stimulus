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
    NAMED WORKFLOW FOR PIPELINE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

include { CHECK_MODEL } from './subworkflows/check_model.nf'
include { HANDLE_DATA } from './workflows/handle_data.nf'
include { HANDLE_TUNE } from './workflows/handle_tune.nf'

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
        params.train_conf
    )
    completion_message = CHECK_MODEL.out.completion_message

    HANDLE_DATA( 
        params.csv,
        params.exp_conf,
        completion_message
    )
    prepared_data = HANDLE_DATA.out.data
    //HANDLE_DATA.out.data.view()

    HANDLE_TUNE(
        params.model,
        params.train_conf,
        prepared_data
    )
    HANDLE_TUNE.out.data.view()
    // HANDLE_ANALYSIS()

}


workflow.onComplete {
    println "Done"
}

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    THE END
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

