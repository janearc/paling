#!/usr/bin/env bash
# Regenerate Python protobuf bindings from the .proto sources.
#
# Generated bindings are intentionally not committed: protoc marks its output
# "NO CHECKED-IN PROTOBUF GENCODE", and committed bindings drift from their
# source. The .proto files under paling/proto/ are the source of truth; run
# this after editing any of them.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if ! command -v protoc >/dev/null 2>&1; then
  echo "protoc not found on PATH; install the protocol buffer compiler" >&2
  exit 1
fi

# protoc ships the well-known types (timestamp.proto, etc.) under its own
# include directory, resolved relative to the binary location.
protoc_prefix="$(cd "$(dirname "$(command -v protoc)")/.." && pwd)"
wkt_include="$protoc_prefix/include"

protoc \
  -I "$repo_root" \
  -I "$wkt_include" \
  --python_out="$repo_root" \
  paling/proto/banchan_event.proto

echo "generated paling/proto/*_pb2.py from .proto sources"
