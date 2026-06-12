#!/usr/bin/env bash
# Fake reducer for tests — echoes a predictably shorter version of stdin.
#
# Simulates a 50% reduction: reads stdin, outputs every other character.
# This gives a known realized ratio of ~0.5 so tests can assert exact values
# without depending on rtk or headroom being installed.
#
# Usage (same interface as `rtk pipe`):
#   echo "some text" | ./fake_reducer.sh
#
# Output: every other byte of stdin, so len(out) ≈ len(in) / 2
python3 -c "
import sys
data = sys.stdin.read()
# Output half the characters (every other one) to simulate ~50% compression.
print(data[::2], end='')
"
