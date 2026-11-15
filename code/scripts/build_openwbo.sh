#!/bin/bash
# Builds Open-WBO (the MaxSAT solver used for all experiments) into code/bin/.
# Requires git, make, g++ and zlib (Debian/Ubuntu: apt install build-essential zlib1g-dev git).
set -e
cd "$(dirname "$0")/.."
mkdir -p bin build
if [ ! -d build/open-wbo ]; then
  git clone https://github.com/sat-group/open-wbo.git build/open-wbo
fi
cd build/open-wbo
git checkout 80f3073e41028b219b0b0ad7c61fba28351f88e6   # version used in the paper (2019-06-13)
make r
cp open-wbo_release ../../bin/
echo "Open-WBO installed in code/bin/open-wbo_release"
