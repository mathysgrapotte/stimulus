# mambaorg/micromamba:1.5.8-bookworm-slim amd64
FROM mambaorg/micromamba@sha256:20a4558dd85eb3e766bfe3eb232461629b186475707123cf721d23b46e075050
LABEL maintainer="Alessio Vignoli" \
        name="alessiovignoli3/stimulus:ps" \
        description="Docker image containing python packages required for stimulus interpet config module"

RUN micromamba install -y -n base -c defaults -c bioconda -c conda-forge \
        python=3.11.8 \
        typing_extensions=4.11.0 \
        importlib_metadata=7.1.0 \
        procps-ng=4.0.4 \
        && micromamba clean -a -y

ENV PATH="$MAMBA_ROOT_PREFIX/bin:$PATH"
USER root