#!/bin/bash
# Serve gpt-oss-20b with /v1/responses endpoint
# RTX 5090 (32GB VRAM)

set -e

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate gpt-oss-vllm

MODEL="openai/gpt-oss-20b"
PORT=8000

# ============================================
# Tool Server Configuration
# ============================================
# Options:
#   - "demo"     : Built-in demo tools (browser + python)
#   - "host:port": External MCP server
#   - ""         : No built-in tools (custom function tools only)
TOOL_SERVER="demo"

# For web_search_preview tool (browser)
# Get API key from: https://exa.ai
export EXA_API_KEY="${EXA_API_KEY:-}"

# For code_interpreter tool (python)
# Options: "dangerously_use_uv", "docker"
export PYTHON_EXECUTION_BACKEND="${PYTHON_EXECUTION_BACKEND:-dangerously_use_uv}"

# Enable response storage for multi-turn conversations
export VLLM_ENABLE_RESPONSES_API_STORE=1

# CRITICAL: Enable system tool labels for gpt-oss to USE the tools
# Without this, the model sees tools but won't actually call them!
export VLLM_GPT_OSS_SYSTEM_TOOL_MCP_LABELS="code_interpreter,web_search_preview,container"

# Help the model follow tool instructions better
export VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS=1

# ============================================
# Model Configuration
# ============================================
# RTX 5090 has 32GB VRAM - gpt-oss-20b should fit comfortably
MAX_MODEL_LEN=16384

# Build args
ARGS=(
    "$MODEL"
    --port "$PORT"
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization 0.9
)

# Add tool server if specified
if [ -n "$TOOL_SERVER" ]; then
    ARGS+=(--tool-server "$TOOL_SERVER")
fi

# Blackwell (RTX 50 series) may need eager mode for stability
# Remove this once CUDA graphs are fully supported
ARGS+=(--enforce-eager)

# Optional: Enable request logging for debugging
# ARGS+=(--enable-log-requests --enable-log-outputs)

echo "============================================"
echo "Starting vLLM server for gpt-oss-20b"
echo "============================================"
echo "Model: $MODEL"
echo "Port: $PORT"
echo "Max Model Length: $MAX_MODEL_LEN"
echo "Tool Server: ${TOOL_SERVER:-disabled}"
echo "Web Search: $([ -n "$EXA_API_KEY" ] && echo 'enabled' || echo 'disabled (set EXA_API_KEY)')"
echo "Code Interpreter: $PYTHON_EXECUTION_BACKEND"
echo "============================================"
echo ""
echo "Endpoints available:"
echo "  - POST /v1/responses      (auto tool execution)"
echo "  - POST /v1/chat/completions (manual tool handling)"
echo "  - GET  /v1/models"
echo "  - GET  /health"
echo "============================================"
echo ""

exec vllm serve "${ARGS[@]}"
