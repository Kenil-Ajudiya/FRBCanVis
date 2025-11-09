#!/bin/bash

# A wrapper script to easily run the 'see' command of can_visualiser.py

TDSOFT=/lustre_archive/apps/tdsoft/
PYTHON_APP="$TDSOFT/FRBCanVis/can_visualiser.py"


# --- Argument Check ---
# Check if at least one argument (the candidate file) is provided.
if [ -z "$1" ]; then
    echo "Usage: $0 <path_to_candidate.h5> [options]"
    echo "------------------------------------------------"
    echo "A simple wrapper to visualise a single FRB candidate."
    echo ""
    echo "Required Argument:"
    echo "  <path_to_candidate.h5>   The full path to the candidate's .h5 file."
    echo ""
    echo "Optional Arguments:"
    echo "  --zoom, -z               Apply a central zoom to the plots."
    echo ""
    echo "Example:"
    echo "  $0 /path/to/data/BM0/123/candidate_xyz.h5 --zoom"
    exit 1
fi

source $TDSOFT/env.sh

# Announce and run the command
echo "Plotting... : python $PYTHON_APP see --cand-file \"$@\""

python "$PYTHON_APP" see --cand-file "$@"

echo "Plotting completed."
