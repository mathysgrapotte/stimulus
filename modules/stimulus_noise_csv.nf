
process STIMULUS_NOISE_CSV {

    container 'alessiovignoli3/stimulus:torch_scikit_numpy'

    input:
    path csv
    path json

    output:
    stdout emit: standardout

    script:
    """
     -c csv -j json
    """
}