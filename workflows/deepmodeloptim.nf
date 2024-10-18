/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    IMPORT MODULES / SUBWORKFLOWS / FUNCTIONS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

//
// SUBWORKFLOW: Consisting of a mix of local and nf-core/modules
//
include { paramsSummaryMap       } from 'plugin/nf-schema'
include { softwareVersionsToYAML } from '../subworkflows/nf-core/utils_nfcore_pipeline'
include { methodsDescriptionText } from '../subworkflows/local/utils_nfcore_deepmodeloptim_pipeline'
include { CHECK_MODEL            } from '../subworkflows/local/check_model'
include { HANDLE_DATA            } from '../subworkflows/local/handle_data'
include { HANDLE_TUNE            } from '../subworkflows/local/handle_tune'
include { HANDLE_ANALYSIS        } from '../subworkflows/local/handle_analysis'

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    RUN MAIN WORKFLOW
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

workflow DEEPMODELOPTIM {

    take:
    // ch_samplesheet, // channel: samplesheet read in from --input
    ch_csv
    ch_exp_conf
    ch_model
    ch_tune_conf
    ch_initial_weights

    main:

    ch_versions = Channel.empty()
    
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

    //
    // Collate and save software versions
    //
    // TODO: collect software versions
    // softwareVersionsToYAML(ch_versions)
    //     .collectFile(
    //         storeDir: "${params.outdir}/pipeline_info",
    //         name: 'nf_core_'  + 'pipeline_software_' +  ''  + 'versions.yml',
    //         sort: true,
    //         newLine: true
    //     ).set { ch_collated_versions }


    emit:
    versions       = ch_versions                 // channel: [ path(versions.yml) ]

}

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    THE END
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/
