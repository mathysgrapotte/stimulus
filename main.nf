#!/usr/bin/env nextflow
/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    nf-core/deepmodeloptim
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Github : https://github.com/nf-core/deepmodeloptim
    Website: https://nf-co.re/deepmodeloptim
    Slack  : https://nfcore.slack.com/channels/deepmodeloptim
----------------------------------------------------------------------------------------
*/

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    VALIDATE & PRINT PARAMETER SUMMARY
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/



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
    //HANDLE_DATA.out.data.view()
    
    HANDLE_TUNE(
        params.model,
        params.tune_conf,
        prepared_data,
        params.initial_weights
    )
    //HANDLE_TUNE.out.model.view()
    //HANDLE_TUNE.out.tune_out.view()
    
    // this part works, but the docker container is not updated with matplotlib yet
    HANDLE_ANALYSIS(
        HANDLE_TUNE.out.tune_out,
        HANDLE_TUNE.out.model
    ) 
    
}

/*
=======================
*/


include { DEEPMODELOPTIM          } from './workflows/deepmodeloptim'
include { PIPELINE_INITIALISATION } from './subworkflows/local/utils_nfcore_deepmodeloptim_pipeline'
include { PIPELINE_COMPLETION     } from './subworkflows/local/utils_nfcore_deepmodeloptim_pipeline'

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    NAMED WORKFLOWS FOR PIPELINE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

//
// WORKFLOW: Run main analysis pipeline depending on type of input
//
workflow NFCORE_DEEPMODELOPTIM {

    take:
    // samplesheet // channel: samplesheet read in from --input
    csv
    exp_conf,
    model,
    tune_conf,
    initial_weights

    main:

    //
    // WORKFLOW: Run pipeline
    //
    DEEPMODELOPTIM (
        // samplesheet,
        csv,
        exp_conf,
        model,
        tune_conf,
        initial_weights
    )
}

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    RUN MAIN WORKFLOW
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

workflow {

    main:
    //
    // SUBWORKFLOW: Run initialisation tasks
    //
    PIPELINE_INITIALISATION (
        params.version,
        params.validate_params,
        params.monochrome_logs,
        args,
        params.outdir //,
        // params.input
    )
    
    //
    // WORKFLOW: Run main workflow
    //
    NFCORE_DEEPMODELOPTIM (
        // PIPELINE_INITIALISATION.out.samplesheet,
        params.csv,
        params.exp_conf,
        params.model,
        params.tune_conf,
        params.initial_weights
    )

    //
    // SUBWORKFLOW: Run completion tasks
    //
    PIPELINE_COMPLETION (
        params.email,
        params.email_on_fail,
        params.plaintext_email,
        params.outdir,
        params.monochrome_logs,
        params.hook_url,
        
    )
}

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    THE END
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/
