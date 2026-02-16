FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel

ARG UNAME=user
ARG PUID
ARG PGID
ARG CODEX_VERSION="latest"
ARG CLAUDE_VERSION="latest"

ENV NVM_VERSION=0.39.7
ENV NODE_VERSION=24.11.1
ENV NVM_DIR="/opt/nvm"
ENV PATH="${NVM_DIR}/versions/node/v${NODE_VERSION}/bin:/home/${UNAME}/.local/bin:${PATH}"

ENV LOGURU_LEVEL=DEBUG 
ENV CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1 
ENV UV_SYSTEM_PYTHON=true
ENV UV_DEV=true
ENV UV_NO_CACHE=true

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

RUN mkdir -p ${NVM_DIR} /var/local/cache/npm /var/local/cache/node-gyp && \
    curl -fsSL -o /tmp/nvm-install.sh "https://raw.githubusercontent.com/nvm-sh/nvm/v${NVM_VERSION}/install.sh" && \
    bash /tmp/nvm-install.sh; rm -f /tmp/nvm-install.sh && \
    . "${NVM_DIR}/nvm.sh"; nvm install "${NODE_VERSION}"
RUN npm i -r @modelcontextprotocol/server-sequential-thinking && uv pip install mcp-server-time
RUN npm i -g "@openai/codex@${CODEX_VERSION}" && \
    mkdir -p /opt/codex
COPY .codex-home /opt/codex
RUN curl -fsSL https://claude.ai/install.sh | bash -s "${CLAUDE_VERSION}" && \
    mkdir -p /home/${UNAME}/.claude
COPY .claude-home /home/${UNAME}/.claude

# Copy project files
WORKDIR /src
COPY ./pyproject.toml ./pyproject.toml
COPY ./README.md ./README.md
COPY ./src ./src

RUN git clone git@github.com:microsoft/markitdown.git && \
    cd markitdown && \
    uv pip install 'packages/markitdown[all]' && \
    uv pip install 'packages/markitdown-mcp' && \
    cd .. && rm -rf markitdown

# Create user and setup permissions
RUN groupadd -g ${PGID} -o ${UNAME} && \
    useradd -m -u ${PUID} -g ${PGID} -o -s /bin/bash ${UNAME}

RUN mkdir -p /data && \
    chown ${PUID}:${PGID} -R /usr/local /data /opt && \
    echo "${UNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${UNAME}

USER ${UNAME}
WORKDIR /src

# Install Python package in editable mode
RUN uv pip install -r pyproject.toml --group dev --group chatterbox && uv pip install -e .

