#!/bin/bash

# Quick launcher for Media Manager
# This script launches the main media manager from the project root

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Launch the media manager
exec "$SCRIPT_DIR/scripts/media-manager.sh"