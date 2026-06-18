#!/usr/bin/env python3
"""Minimal solver for test_crypto_01 (E2E fixture)."""

import hashlib

print(hashlib.sha256(b"flag").hexdigest())
