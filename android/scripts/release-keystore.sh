#!/usr/bin/env bash
# Generate a release signing keystore and print the GitHub secrets to add.
# Obtainium requires the same signing key across versions, so commit none of
# this — store the keystore offline + put the secrets in GitHub.
set -euo pipefail

OUT="${1:-booki-release.keystore}"
ALIAS="${KEY_ALIAS:-booki}"

if [ -e "$OUT" ]; then
    echo "Refusing to overwrite existing $OUT" >&2
    exit 1
fi

read -rsp "Keystore password: " KS_PASS; echo
read -rsp "Key password (often same as keystore): " KEY_PASS; echo

keytool -genkeypair -v \
    -keystore "$OUT" -alias "$ALIAS" \
    -keyalg RSA -keysize 4096 -validity 36500 \
    -storepass "$KS_PASS" -keypass "$KEY_PASS" \
    -dname "CN=Booki, O=Booki, C=US"

echo
echo "Add the following GitHub repo secrets (Settings → Secrets and variables → Actions):"
echo "  KEYSTORE_BASE64    $(base64 -w0 < "$OUT")"
echo "  KEYSTORE_PASSWORD  <the keystore password>"
echo "  KEY_ALIAS          $ALIAS"
echo "  KEY_PASSWORD       <the key password>"
echo
echo "Keep $OUT in a safe place. Losing it means Obtainium users cannot"
echo "upgrade — they would have to uninstall + reinstall."
