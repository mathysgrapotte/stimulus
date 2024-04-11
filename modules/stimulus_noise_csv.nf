
process STIMULUS_NOISE_CSV {

    container 'alessiovignoli3/stimulus:torch_scikit_numpy'

    input:
    tuple path(csv), val(user_json), path(parsed_json)

    output:
    stdout emit: standardout

    script:
    """
    launch_noise_csv.py -c ${csv} -j ${parsed_json}
    """
}