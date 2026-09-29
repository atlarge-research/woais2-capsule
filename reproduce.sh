#!/usr/bin/env bash
# Regenerates the paper's five figures into out/: creates a Python virtual environment in .venv/, installs the pinned
# packages into it and runs plot.py.
#
# Needs Python 3.11 (python3.11 on the PATH, or PYTHON=/path/to/python3.11 ./reproduce.sh): the pinned pyarrow 12.0.1
# has no wheels for later versions, and matplotlib 3.11 none for earlier ones.
# Needs the font Times New Roman, regular and bold, in which the figures are set: macOS ships it; on Debian/Ubuntu,
# install ttf-mscorefonts-installer (then delete .venv/matplotlib, matplotlib's list of the fonts it found).
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
    PYTHON="${PYTHON:-python3.11}"
    "$PYTHON" -c 'import sys; sys.exit(sys.version_info[:2] != (3, 11))' 2> /dev/null \
        || { echo "needs Python 3.11: PYTHON=/path/to/python3.11 ./reproduce.sh" >&2; exit 1; }
    # from the resolved interpreter: a venv made through a symlink to it can lose its standard library
    "$("$PYTHON" -c 'import os, sys; print(os.path.realpath(sys.executable))')" -m venv .venv
fi

# The exact versions that drew the paper's figures, with every package they pull in. matplotlib 3.11.0: 3.10 places
# the text up to 1.7 pt elsewhere. datasets 2.13.2: the last release that ignores the dataset card's `configs`, which
# name its only split "all", a name that datasets >= 2.14 rejects.
PACKAGES=(
    matplotlib==3.11.0 contourpy==1.3.2 cycler==0.12.1 fonttools==4.65.0 kiwisolver==1.5.1 packaging==26.3
    pillow==12.3.0 pyparsing==3.3.3 python-dateutil==2.9.0.post0 six==1.17.0
    numpy==1.26.4 pandas==2.3.3 pytz==2026.4 tzdata==2026.4
    datasets==2.13.2 aiohappyeyeballs==2.7.1 aiohttp==3.14.3 aiosignal==1.4.0 async-timeout==5.0.1 attrs==26.1.0
    certifi==2026.7.22 charset-normalizer==3.5.1 dill==0.3.6 filelock==4.0.3 frozenlist==1.8.0 fsspec==2023.6.0
    huggingface-hub==0.16.4 idna==3.20 multidict==6.9.1 multiprocess==0.70.14 propcache==0.5.4 pyarrow==12.0.1
    PyYAML==6.0.3 requests==2.34.2 tqdm==4.70.1 typing_extensions==4.16.0 urllib3==2.8.0 xxhash==4.0.1 yarl==1.25.1
)
echo "installing the pinned packages into .venv/"
.venv/bin/python -m pip install --quiet --disable-pip-version-check "${PACKAGES[@]}"

# matplotlib's font list and the downloaded dataset stay in .venv/ as well
MPLCONFIGDIR="$PWD/.venv/matplotlib" HF_HOME="$PWD/.venv/huggingface" .venv/bin/python plot.py
