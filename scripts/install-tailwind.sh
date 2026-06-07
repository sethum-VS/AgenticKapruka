#!/usr/bin/env bash
# Download Tailwind CSS standalone CLI (no Node.js required).
set -euo pipefail

VERSION="v3.4.17"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BIN_DIR="${ROOT_DIR}/bin"
TARGET="${BIN_DIR}/tailwindcss"

mkdir -p "${BIN_DIR}"

OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"

case "${OS}-${ARCH}" in
  darwin-arm64) ASSET="tailwindcss-macos-arm64" ;;
  darwin-x86_64) ASSET="tailwindcss-macos-x64" ;;
  linux-x86_64) ASSET="tailwindcss-linux-x64" ;;
  linux-aarch64 | linux-arm64) ASSET="tailwindcss-linux-arm64" ;;
  *)
    echo "Unsupported platform: ${OS} ${ARCH}" >&2
    exit 1
    ;;
esac

URL="https://github.com/tailwindlabs/tailwindcss/releases/download/${VERSION}/${ASSET}"
echo "Downloading Tailwind CSS ${VERSION} (${ASSET})..."
curl -fsSL "${URL}" -o "${TARGET}"
chmod +x "${TARGET}"
echo "Installed ${TARGET}"
