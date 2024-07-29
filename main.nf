#!/usr/bin/env nextflow
/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    VALIDATE & PRINT PARAMETER SUMMARY
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

include { validateParameters; paramsHelp } from 'plugin/nf-validation'

// Print help message if needed
if (params.help) {
    log.info paramsHelp("nextflow run main.nf --csv my_data.csv --exp_conf exp.yaml --model my_model.py --tune_conf tune.yaml -profie your_profile ")
    System.exit(0)
}

// Validate input parameters
if (params.validate_params) {
    validateParameters()
}

WorkflowMain.initialise(workflow, params, log)

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
        params.tune_conf
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
        params.tune_conf,
        prepared_data
    )
    //HANDLE_TUNE.out.model.view()
    //HANDLE_TUNE.out.tune_out.view()
    
    // this part works, but the docker container is not updated with matplotlib yet
    HANDLE_ANALYSIS(
        HANDLE_TUNE.out.tune_out,
        HANDLE_TUNE.out.model
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

