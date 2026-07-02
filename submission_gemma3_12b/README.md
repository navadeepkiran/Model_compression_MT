# WMT26 Model Compression Submission

**Model Name**: Gemma-3 12B (Pruned to 40 Layers & 70% MLP Width)
**Hugging Face Repository**: `nani-nav/gemma-3-12b-final-wmt-4488`
**Quantization**: BitsAndBytes INT4 (NF4, Double Quantization)

## Execution Environment
- Environment preparation script: `setup.sh`
- Dependencies: `requirements.txt`
- Inference script: `run.sh` (wraps `inference.py`)

## Inference Details
The model uses `transformers` and `bitsandbytes` to load the pruned Gemma-3 12B model in INT4. 
The script parses WMT-formatted inputs strictly line-by-line, and outputs strictly one translation line per input line.
All logging is deliberately sent to `stderr` to prevent evaluation parsing failures.

## Memory and Speed
- Max Batch Size limit: 8 (to prevent OOM on single GPU constraints)
- Precision: 4-bit NormalFloat (NF4)
