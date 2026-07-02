#!/usr/bin/env bash
# WMT26 Inference Entry Point
# Allows both explicit and positional arguments

# Determine the directory where this script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate the virtual environment
if [ -f "$DIR/.venv/bin/activate" ]; then
    source "$DIR/.venv/bin/activate"
else
    echo "Warning: .venv not found. Did you run setup.sh?" >&2
fi

# Explicit form
if [ "$1" == "--lang-pair" ]; then
    LANG_PAIR=$2
    BATCH_SIZE=$4
    INPUT=$6
    OUTPUT=$8
    
    python "$DIR/inference.py" \
      --lang_pair "$LANG_PAIR" \
      --batch_size "$BATCH_SIZE" \
      --input_file "$INPUT" \
      --output_file "$OUTPUT"

# Positional form
else
    LANG_PAIR=$1
    BATCH_SIZE=$2
    
    # Run the Python script, streaming stdin to it, and letting it output to stdout
    python "$DIR/inference.py" \
      --lang_pair "$LANG_PAIR" \
      --batch_size "$BATCH_SIZE" \
      --use_stdin
fi
