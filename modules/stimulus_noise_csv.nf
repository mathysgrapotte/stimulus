
process STIMULUS_NOISE_CSV {

    container 'alessiovignoli3/stimulus:torch_scikit_numpy'

    input:
    tuple path(csv), val(user_json), path(parsed_json)

    output:
    stdout emit: standardout

    script:
    """
    echo ${csv} ${user_json} ${parsed_json}
    """
}