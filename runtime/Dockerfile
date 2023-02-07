FROM nvidia/cuda:11.0.3-base-ubuntu20.04
# This CUDA version does not matter
# We will separately install the CUDA runtime via cudatoolkit using conda

ARG CPU_OR_GPU=gpu

ENV CONDA_VERSION=4.10.3-2 \
    CONDA_ENV=condaenv \
    RUNTIME_USER=appuser \
    RUNTIME_UID=1000 \
    SHELL=/bin/bash \
    LANG=C.UTF-8  \
    LC_ALL=C.UTF-8 \
    CONDA_DIR=/opt/conda \
    DEBIAN_FRONTEND=noninteractive

ENV CPU_OR_GPU=${CPU_OR_GPU} \
    HOME=/home/${RUNTIME_USER} \
    PATH=${CONDA_DIR}/bin:${PATH}

# ======================== root ========================
# initialize paths we will use
RUN mkdir -p /code_execution

# Create appuser user, permissions, add conda init to startup script
RUN echo "Creating ${RUNTIME_USER} user..." \
    && groupadd --gid ${RUNTIME_UID} ${RUNTIME_USER}  \
    && useradd --create-home --gid ${RUNTIME_UID} --no-log-init --uid ${RUNTIME_UID} ${RUNTIME_USER} \
    && echo ". ${CONDA_DIR}/etc/profile.d/conda.sh ; conda activate ${CONDA_ENV}" > /etc/profile.d/init_conda.sh \
    && chown -R ${RUNTIME_USER}:${RUNTIME_USER} /opt /code_execution

# Install base packages
ARG DEBIAN_FRONTEND=noninteractive
COPY ./apt.txt /home/${RUNTIME_USER}
RUN rm -f /etc/apt/sources.list.d/cuda.list && rm -f /etc/apt/sources.list.d/nvidia-ml.list
RUN echo "Installing base packages..." \
    && apt-get update --fix-missing \
    && apt-get install -y apt-utils 2> /dev/null \
    && apt-get install -y wget zip tzdata \
    && apt-get install -y libgl1-mesa-glx \
    && xargs -a /home/${RUNTIME_USER}/apt.txt apt-get install -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /home/${RUNTIME_USER}/apt.txt
# ======================================================

# ======================== user ========================
USER ${RUNTIME_USER}

# Install conda
RUN echo "Installing Miniforge..." \
    && URL="https://github.com/conda-forge/miniforge/releases/download/${CONDA_VERSION}/Miniforge3-${CONDA_VERSION}-Linux-x86_64.sh" \
    && wget --quiet ${URL} -O /home/${RUNTIME_USER}/miniconda.sh \
    && /bin/bash /home/${RUNTIME_USER}/miniconda.sh -u -b -p ${CONDA_DIR} \
    && rm /home/${RUNTIME_USER}/miniconda.sh \
    && conda install -y -c conda-forge mamba \
    && mamba clean -afy \
    && find ${CONDA_DIR} -follow -type f -name '*.a' -delete \
    && find ${CONDA_DIR} -follow -type f -name '*.pyc' -delete

# Switch back to root for installing conda packages
COPY environment-${CPU_OR_GPU}.yml /home/${RUNTIME_USER}/environment.yml
RUN [ ${CPU_OR_GPU} = "gpu" ] && export CONDA_OVERRIDE_CUDA="11.0" || export CONDA_OVERRIDE_CUDA= \
    && mamba env create --name ${CONDA_ENV} -f /home/${RUNTIME_USER}/environment.yml  \
    && mamba clean -afy \
    && conda run pip cache purge \
    && rm /home/${RUNTIME_USER}/environment.yml \
    && find ${CONDA_DIR} -follow -type f -name '*.a' -delete \
    && find ${CONDA_DIR} -follow -type f -name '*.pyc' -delete \
    && find ${CONDA_DIR} -follow -type f -name '*.js.map' -delete

# Copy run script into working dir and set it as the working doie
WORKDIR /code_execution
RUN mkdir -p data predictions package /home/${RUNTIME_USER}/.config/procps
COPY --chown=appuser:appuser tests /code_execution/tests
COPY --chown=appuser:appuser supervisor.py /code_execution/supervisor.py
COPY --chown=appuser:appuser main_federated_train.py /code_execution/main_federated_train.py
COPY --chown=appuser:appuser main_federated_test.py /code_execution/main_federated_test.py
COPY --chown=appuser:appuser post_federated.py /code_execution/post_federated.py
COPY --chown=appuser:appuser main_centralized_train.py /code_execution/main_centralized_train.py
COPY --chown=appuser:appuser main_centralized_test.py /code_execution/main_centralized_test.py
COPY --chown=appuser:appuser post_centralized.py /code_execution/post_centralized.py
COPY --chown=appuser:appuser entrypoint.sh /code_execution/entrypoint.sh
COPY --chown=appuser:appuser toprc /home/${RUNTIME_USER}/.config/procps/toprc

ENTRYPOINT ["/bin/bash", "/code_execution/entrypoint.sh"]
