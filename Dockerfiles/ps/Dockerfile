# mambaorg/micromamba:1.5.8-bookworm-slim
FROM mambaorg/micromamba@sha256:abcb3ae7e3521d08e1fdeaff63131765b34e4f29b6a8a2c28660036b53841569
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