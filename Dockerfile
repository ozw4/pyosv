# syntax=docker/dockerfile:1

FROM nvcr.io/nvidia/pytorch:24.09-py3 AS develop

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG USERNAME=dcuser
ARG UID=1000
ARG GID=1000
ARG CODEX_VERSION=latest

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

ENV USER_UID=${UID} \
    USER_GID=${GID} \
    HTTP_PROXY=${HTTP_PROXY} \
    HTTPS_PROXY=${HTTPS_PROXY} \
    NO_PROXY=${NO_PROXY} \
    http_proxy=${HTTP_PROXY} \
    https_proxy=${HTTPS_PROXY} \
    no_proxy=${NO_PROXY} \
    DEBIAN_FRONTEND=noninteractive

RUN echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" | debconf-set-selections

RUN --mount=type=cache,target=/var/lib/apt,sharing=locked \
    --mount=type=cache,target=/var/cache/apt,sharing=locked \
    set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        fontconfig \
        git \
        gpg \
        shellcheck \
        ttf-mscorefonts-installer; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*

RUN fc-cache -fv

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python -m pip install torchaudio==2.7.0

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    --mount=type=bind,source=.devcontainer/requirements-dev.txt,target=/tmp/requirements-dev.txt,readonly \
    python -m pip install -r /tmp/requirements-dev.txt

RUN set -eux; \
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
        | gpg --dearmor --yes -o /usr/share/keyrings/githubcli-archive-keyring.gpg; \
    chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg; \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
        > /etc/apt/sources.list.d/github-cli.list; \
    apt-get update; \
    apt-get install -y --no-install-recommends gh; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -; \
    apt-get update; \
    apt-get install -y --no-install-recommends nodejs; \
    node --version; \
    npm --version; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    groupadd --gid "${USER_GID}" "${USERNAME}"; \
    useradd --uid "${USER_UID}" --gid "${USER_GID}" --create-home --shell /bin/bash "${USERNAME}"; \
    install -d -m 0755 -o "${USERNAME}" -g "${USERNAME}" \
        "/home/${USERNAME}/.local" \
        "/home/${USERNAME}/.local/bin" \
        "/home/${USERNAME}/.npm" \
        "/workspace"; \
    install -d -m 0700 -o "${USERNAME}" -g "${USERNAME}" \
        "/home/${USERNAME}/.codex" \
        "/home/${USERNAME}/.codex/tmp" \
        "/home/${USERNAME}/.codex/tmp/arg0"; \
    chown -R "${USERNAME}:${USERNAME}" "/home/${USERNAME}" "/workspace"

USER ${USERNAME}

ENV HOME=/home/${USERNAME} \
    CODEX_HOME=/home/${USERNAME}/.codex \
    NPM_CONFIG_PREFIX=/home/${USERNAME}/.local \
    PATH="/home/${USERNAME}/.local/bin:${PATH}" \
    PYTHONPATH="/workspace"

RUN set -eux; \
    npm config set prefix "${NPM_CONFIG_PREFIX}"; \
    npm install -g "@openai/codex@${CODEX_VERSION}"; \
    codex --version; \
    npm cache clean --force

RUN set -eux; \
    curl -LsSf https://astral.sh/uv/install.sh | sh; \
    uv --version

RUN set -eux; \
    uv tool install specify-cli --from git+https://github.com/github/spec-kit.git@v0.4.4

COPY --chown=${USERNAME}:${USERNAME} ruff.toml /home/${USERNAME}/

WORKDIR /workspace

CMD ["/bin/bash"]
