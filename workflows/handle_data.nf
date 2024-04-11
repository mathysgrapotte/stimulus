/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    IMPORT NF-CORE MODULES/SUBWORKFLOWS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

include { INTERPRET_JSON } from '../modules/interpret_json.nf'
include { NOISE_CSV      } from '../subworkflows/noise_csv.nf'

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

    // launch the actual noise subworkflow
    NOISE_CSV(csv, json_tuple)


    emit:

    debug = NOISE_CSV.out.debug

}


/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    THE END
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/