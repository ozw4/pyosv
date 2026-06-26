# syntax=docker/dockerfile:1

FROM nvcr.io/nvidia/pytorch:24.09-py3 AS develop

ARG USERNAME=dcuser
ARG UID=1000
ARG GID=1000

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

ENV HTTP_PROXY=${HTTP_PROXY} \
    HTTPS_PROXY=${HTTPS_PROXY} \
    NO_PROXY=${NO_PROXY} \
    DEBIAN_FRONTEND=noninteractive \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

RUN echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" | debconf-set-selections

RUN --mount=type=cache,target=/var/lib/apt,sharing=locked \
    --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    fontconfig \
    git \
    gpg \
    shellcheck \
    ttf-mscorefonts-installer \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN fc-cache -fv

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install torchaudio==2.7.0

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    --mount=type=bind,source=.devcontainer/requirements-dev.txt,target=requirements-dev.txt \
    python -m pip install -r requirements-dev.txt

RUN mkdir -p "$PLAYWRIGHT_BROWSERS_PATH" \
    && python -m playwright install --with-deps chromium
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
    | gpg --dearmor -o /usr/share/keyrings/githubcli-archive-keyring.gpg \
 && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
 && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
    > /etc/apt/sources.list.d/github-cli.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends gh \
 && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
 && apt-get update \
 && apt-get install -y --no-install-recommends nodejs \
 && npm install -g @openai/codex@latest \
 && codex --version \
 && npm cache clean --force \
 && rm -rf /var/lib/apt/lists/*

RUN addgroup --gid $GID $USERNAME && \
    adduser --disabled-password --gecos "" --shell "/bin/bash" --uid $UID --gid $GID $USERNAME && \
    mkdir -p /home/$USERNAME/.codex && \
    chown -R $USERNAME:$USERNAME /home/$USERNAME "$PLAYWRIGHT_BROWSERS_PATH"

USER $USERNAME

ENV HOME=/home/$USERNAME \
    CODEX_HOME=/home/$USERNAME/.codex \
    PATH="/home/$USERNAME/.local/bin:${PATH}" \
    PYTHONPATH="/workspace"

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN uv tool install specify-cli --from git+https://github.com/github/spec-kit.git@v0.4.4

COPY --chown=$USERNAME:$USERNAME ruff.toml /home/$USERNAME/
WORKDIR /workspace

CMD ["/bin/bash"]
