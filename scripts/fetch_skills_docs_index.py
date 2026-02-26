#!/usr/bin/env python3
from __future__ import annotations

import urllib.request

URL = "https://agentskills.io/llms.txt"

if __name__ == "__main__":
    with urllib.request.urlopen(URL, timeout=30) as response:
        print(response.read().decode("utf-8"))
