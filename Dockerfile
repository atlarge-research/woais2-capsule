# The reproducibility capsule in a container: `make docker` builds this image and runs it with ./out mounted.
FROM python:3.11-slim

# Same settings as the Makefile. Agg is load-bearing: other matplotlib backends write narrower PDFs than declared.
# Caches live in /tmp so that the container can run as the host user (docker run --user), who owns ./out.
ENV MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    HF_HOME=/tmp/huggingface \
    DATASETS_VERBOSITY=error \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /capsule
COPY requirements.txt .
RUN pip install --quiet -r requirements.txt
COPY reproduce.py figures.py check.py ./

# Downloads the public dataset from the Hugging Face Hub (needs network), writes out/, then checks it against the paper.
CMD ["sh", "-c", "python reproduce.py && python check.py"]
