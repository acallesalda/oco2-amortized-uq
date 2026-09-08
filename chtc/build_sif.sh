#!/bin/bash
# Build the OCO-2 Apptainer image. Run this INSIDE the interactive build job:
#
#     condor_submit -i build-container.sub
#     ./build_sif.sh
#     exit
#
# Takes 15-30 minutes. Most of it is pip pulling ~2.5 GB of torch + vendored CUDA wheels.
set -euo pipefail

SIF_NAME="oco2-gpu.sif"
DEF_NAME="oco2-gpu.def"

echo "=============================================================="
echo " OCO-2 container build"
echo " host      : $(hostname)"
echo " scratch   : $PWD"
echo " apptainer : $(apptainer --version 2>/dev/null || echo 'NOT FOUND')"
echo " free disk : $(df -h . | awk 'NR==2 {print $4}')"
echo "=============================================================="

if ! command -v apptainer >/dev/null 2>&1; then
    echo "ERROR: apptainer is not on PATH. You are probably not on a build node."
    echo "       Exit, and check that build-container.sub still has +IsBuildJob = true."
    exit 1
fi

# Keep the build cache inside the job scratch directory so it is cleaned up on exit and
# never lands in /home. HTCondor normally sets this already; belt and braces.
export APPTAINER_CACHEDIR="$PWD/.apptainer-cache"
export APPTAINER_TMPDIR="$PWD/.apptainer-tmp"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

# `%files` cannot be conditional, and a missing source file fails the whole build, so the
# def always copies requirements.lock.txt. When there is no real lock -- the usual case,
# since it can only be captured from a running container -- ship an empty one. The
# def tests for a NON-EMPTY file, so an empty placeholder means "use requirements.txt".
if [ ! -f requirements.lock.txt ]; then
    : > requirements.lock.txt
    echo "note: no requirements.lock.txt; building from requirements.txt constraints."
    echo "      (see the reproducibility note in requirements.txt)"
else
    echo "using requirements.lock.txt ($(wc -l < requirements.lock.txt) pinned packages)"
fi

# A /tmp that the build can actually write to.
#
# Apptainer bind-mounts the host /tmp into the build sandbox. Under an unprivileged
# (user-namespace) build, processes that drop privileges inside the container land on an
# unmapped uid and cannot write there -- which is what kills `apt-get update` with
# "Couldn't create temporary file /tmp/apt.conf.XXXXXX ... repository is not signed" and
# then "%post section: exit status 100". The def file fixes the apt-specific cause; this
# fixes the general one, for anything else in %post that wants /tmp.
BUILD_TMP="$PWD/.build-tmp"
mkdir -p "$BUILD_TMP"
chmod 1777 "$BUILD_TMP"

BUILD_ARGS=(--force)
# `apptainer build` did not always accept --bind. Ask before using it rather than losing
# a 30-minute job to an unknown-flag error.
if apptainer build --help 2>&1 | grep -q -- "--bind"; then
    BUILD_ARGS+=(--bind "$BUILD_TMP:/tmp")
    echo "binding $BUILD_TMP onto /tmp for the build"
else
    echo "note: this apptainer's 'build' has no --bind; relying on the def file's"
    echo "      APT::Sandbox::User setting alone."
fi

echo
echo "=== building $SIF_NAME from $DEF_NAME ==="
apptainer build "${BUILD_ARGS[@]}" "$SIF_NAME" "$DEF_NAME"

echo
echo "=== self-test (CPU only: build nodes have no GPU, so cuda.is_available() is False here) ==="
apptainer exec "$SIF_NAME" python3 - <<'PY'
import torch, numpy, pandas, xarray, hydra, nflows, laplace, h5py, scipy
print("torch      ", torch.__version__)
print("numpy      ", numpy.__version__)
print("pandas     ", pandas.__version__)
print("xarray     ", xarray.__version__)
print("nflows     ", getattr(nflows, "__version__", "(no __version__)"))
print("h5py       ", h5py.__version__)
print("scipy      ", scipy.__version__)
# The two imports most likely to break: native DataTree, and the Laplace backend.
assert hasattr(xarray, "DataTree"), "xarray is too old for xr.DataTree"
from xarray.backends.api import open_datatree          # noqa: F401
from laplace import Laplace                            # noqa: F401
from nflows import flows, distributions, transforms    # noqa: F401
print("OK: DataTree, Laplace and nflows all import")
PY

SIZE_BYTES=$(stat -c %s "$SIF_NAME")
echo
echo "=== built: $SIF_NAME ($(numfmt --to=iec "$SIZE_BYTES" 2>/dev/null || echo "$SIZE_BYTES bytes")) ==="

# ---------------------------------------------------------------------------------
# Ship it to /staging, where the training jobs will pick it up over OSDF.
# ---------------------------------------------------------------------------------
NETID=$(whoami)
STAGING="/staging/${NETID:0:1}/${NETID}/oco2"

if [ -d "$STAGING" ] && [ -w "$STAGING" ]; then
    echo "=== copying to $STAGING ==="
    cp "$SIF_NAME" "$STAGING/$SIF_NAME.partial"
    mv "$STAGING/$SIF_NAME.partial" "$STAGING/$SIF_NAME"
    STAGED_BYTES=$(stat -c %s "$STAGING/$SIF_NAME")
    if [ "$STAGED_BYTES" -eq "$SIZE_BYTES" ]; then
        echo "verified: $STAGING/$SIF_NAME is $STAGED_BYTES bytes"
        # build-container.sub declares oco2-gpu.sif as an output file, so it has to
        # still exist when the job exits -- but there is no point shipping 5 GB back to
        # /home now that /staging has it. Empty the local copy and let HTCondor transfer
        # nothing. If you see a 0-byte oco2-gpu.sif in your submit directory afterwards,
        # that is this line, and it means the staging copy is good.
        : > "$SIF_NAME"
        echo
        echo "DONE. Container is at $STAGING/$SIF_NAME"
        echo "      Type 'exit' to end the interactive job."
    else
        echo "ERROR: staged copy is $STAGED_BYTES bytes, expected $SIZE_BYTES. Not truncating."
        exit 1
    fi
else
    echo "NOTE: $STAGING is not writable from this build node."
    echo "      Leaving $SIF_NAME here; HTCondor will transfer it back to your submit"
    echo "      directory when you exit. Then, from the access point, run:"
    echo
    echo "          mkdir -p $STAGING"
    echo "          mv ~/OCO-2/chtc/$SIF_NAME $STAGING/"
    echo
    echo "      Type 'exit' to end the interactive job."
fi
