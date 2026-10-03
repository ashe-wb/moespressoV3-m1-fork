#!/bin/zsh
# Build steadfastgaze/mlx-kquant at the commit pinned in pyproject.toml with this
# fork's kernel patch applied, for use through m1/moespresso-m1.
#
# Usage: m1/build-mlx-kquant.sh [DEST]   (default: m1/build/mlx-kquant)
# Requires git, uv and the Xcode command line tools with the Metal toolchain.
set -eu
HERE=${0:A:h}
REPO=${HERE:h}
DEST=${1:-$HERE/build/mlx-kquant}
PIN=c4928ba74b6b119c3613e1ec68ef85d2238c636b
PATCH=$HERE/mlx-kquant/0001-Widen-Qwen4-gated-residual-norm-and-front-kernels.patch

grep -q "mlx-kquant.git@$PIN" "$REPO/pyproject.toml" || {
  echo "build-mlx-kquant: pyproject.toml no longer pins mlx-kquant at $PIN" >&2
  exit 1
}
if [[ ! -d $DEST/.git ]]; then
  git clone https://github.com/steadfastgaze/mlx-kquant.git "$DEST"
fi
if git -C "$DEST" apply --reverse --check "$PATCH" 2>/dev/null; then
  echo "build-mlx-kquant: patch already applied in $DEST"
else
  git -C "$DEST" checkout --quiet --detach "$PIN"
  git -C "$DEST" apply "$PATCH"
fi

# The extension must match the Python version of the MoEspresso environment.
PY=$(uv run --locked --project "$REPO" python -c 'import sys; print("%d.%d" % sys.version_info[:2])')
cd "$DEST"
uv run --no-project --python "$PY" \
  --with "setuptools>=77" --with "cmake>=3.27" --with "mlx==0.31.2" --with "nanobind==2.12.0" \
  python setup.py build_ext --inplace -j 8
ls mlx_kquant/_ext.cpython-*-darwin.so mlx_kquant/libmlx_kquant_ext.dylib mlx_kquant/mlx_kquant.metallib
echo "build-mlx-kquant: built $DEST"
