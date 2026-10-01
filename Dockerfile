# Docker image of the capsule: Python 3.11, the font Times New Roman and the pinned packages, all included.
#   docker build -t woais2 .
#   docker run --rm -v "$PWD/out:/capsule/out" woais2
FROM python:3.11.16-slim-trixie@sha256:e41613d42d4891e4930f79523f93f81bbc7632584ec65e36ab055f41a800b41e

# Times New Roman: Microsoft's core fonts, from Debian's contrib section
RUN sed -i 's/^Components: main$/Components: main contrib/' /etc/apt/sources.list.d/debian.sources \
    && echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" | debconf-set-selections \
    && apt-get update \
    && apt-get install -y --no-install-recommends ttf-mscorefonts-installer fontconfig \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /capsule
COPY reproduce.sh plot.py ./
COPY data/ data/
# creates .venv/ with the pinned packages (and draws the figures once, which also checks the image); the caches stay
# writable so that the image also runs as a non-root user (docker run --user)
RUN PIP_NO_CACHE_DIR=1 PYTHON=python3.11 bash reproduce.sh \
    && chmod -R a+rwX .venv/matplotlib .venv/huggingface
ENV HOME=/tmp

CMD ["bash", "reproduce.sh"]
