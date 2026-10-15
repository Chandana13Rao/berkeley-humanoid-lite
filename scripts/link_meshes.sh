#!/usr/bin/env bash
# Recreate the mesh symlinks the MJCF files expect in the assets submodule.
# The XMLs use meshdir="assets" and file="merged/...",
# so mjcf/assets/merged/x.stl -> meshes/x.stl.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data="$repo_root/source/berkeley_humanoid_lite_assets/data"
robot="robots/berkeley_humanoid/berkeley_humanoid_lite"

ln -sfn "$robot/mjcf" "$data/mjcf"
ln -sfn ..             "$data/$robot/mjcf/assets"
ln -sfn meshes         "$data/$robot/merged"

echo "Mesh symlinks created in $data"
