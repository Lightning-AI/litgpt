#!/bin/bash

# Kill all GPU processes
kill -9 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -o '[0-9]\+')