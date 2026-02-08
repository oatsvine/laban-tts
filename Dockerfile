
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel

ARG UNAME=user
ARG PUID
ARG PGID

ENV LOGURU_LEVEL=DEBUG \
    CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1 \
    UV_SYSTEM_PYTHON=true

# Install system packages
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        jq \
        rsync \
        curl \
        less \
        coreutils \
        build-essential \
        ripgrep \
        fd-find \
        neovim \
        uuid-dev \
        sudo \
        zip && \
    rm -rf /var/lib/apt/lists/* && \
    ln -s /usr/bin/fdfind /usr/local/bin/fd

RUN conda install -y conda-forge::uv anaconda::numpy==1.25.2 

# Copy project files
WORKDIR /usr/local/src
COPY ./pyproject.toml ./pyproject.toml
COPY ./README.md ./README.md
COPY ./src ./src

# Create user and setup permissions
RUN groupadd -g ${PGID} -o ${UNAME} && \
    useradd -m -u ${PUID} -g ${PGID} -o -s /bin/bash ${UNAME}

RUN mkdir -p /data && \
    chown ${PUID}:${PGID} -R /usr/local /data /opt && \
    echo "${UNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${UNAME}

USER ${UNAME}
WORKDIR /usr/local/src

# Install Python package in editable mode
RUN uv sync --no-cache

# Run optional post-install script
RUN echo "${POST_INSTALL_SCRIPT}" | bash -s
