FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates git openssh-client sudo bash jq iputils-ping \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for development
RUN useradd -m -s /bin/bash coder \
    && echo "coder ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/coder \
    && chmod 0440 /etc/sudoers.d/coder

USER coder
WORKDIR /home/coder
ENV HOME=/home/coder

# Install OpenCode (official installer)
RUN curl -fsSL https://opencode.ai/install | bash

ENV PATH=/home/coder/.opencode/bin:$PATH

CMD ["bash"]
