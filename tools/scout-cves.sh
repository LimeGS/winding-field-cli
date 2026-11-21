#!/usr/bin/env bash
set -euo pipefail

image="${1:?usage: $0 IMAGE}"
docker scout cves "$image" --only-fixed --only-severity critical,high --exit-code
