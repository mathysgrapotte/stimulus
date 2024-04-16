/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    IMPORT NF-CORE MODULES/SUBWORKFLOWS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

include { INTERPRET_JSON } from '../modules/interpret_json.nf'
include { NOISE_CSV      } from '../subworkflows/noise_csv.nf'
include { SPLIT_CSV      } from '../subworkflows/split_csv.nf'

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    RUN MAIN WORKFLOW
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

workflow HANDLE_DATA {

    take:

    input_csv
    input_json


    main:

    // put the files in channels
    csv  = Channel.fromPath( input_csv  )
    json = Channel.fromPath( input_json )

    // read the json and create many json as there are combinations of noisers and splitters
    INTERPRET_JSON(json)

    // parse the tuple so that each interpreted_json is associated to the name of the User Json that created it
    json_tuple = INTERPRET_JSON.out.interpreted_json.transpose()
    
    // launch splitting subworkflow 
    SPLIT_CSV(csv, json_tuple)

    // make output of splitting subworkflow formatted for the noise subworkflow
// SPLIT_CSV.out.split_data
//         .multiMap{
//         csv_name,csv_split,json_name,json ->
//         csv:  csv_split
//         json_tuple: [json_name,json]
//         }.set{split_data}


    // launch the actual noise subworkflow
    NOISE_CSV( SPLIT_CSV.out.split_data )


    emit:

    debug = NOISE_CSV.out.debug
    data  = NOISE_CSV.out.noised_data

}


/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    THE END
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/
