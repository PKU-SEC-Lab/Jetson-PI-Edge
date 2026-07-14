#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
NSYS_DIR="${PROJECT_DIR}/nsys_repport"
NSYS_BIN="/usr/local/bin/nsys"
CUDA_HOME_DEFAULT="${CUDA_HOME:-/usr/local/cuda}"

MODE=""
USE_NSYS=0
# CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-1}"
CUDA_DEVICE="0"
HOST="127.0.0.1"
PORT="8081"
MODEL_PATH="${MODEL_PATH:-PATH/TO/pi_llm.gguf}"
SERVER_MODEL_PATH="${SERVER_MODEL_PATH:-${MODEL_PATH}}"
MMPROJ_PATH="${MMPROJ_PATH:-PATH/TO/mmproj-model-f16.gguf}"
PI0_ACTION_NOISE_BIN_DEFAULT="${PI0_ACTION_NOISE_BIN_DEFAULT:-PATH/TO/action_noise.bin}"
PI0_NORM_STATS_JSON_DEFAULT="${PI0_NORM_STATS_JSON_DEFAULT:-PATH/TO/norm_stats.json}"
PI05_DEBUG_HOST_LOG_FILE_DEFAULT="/tmp/pi05_llama_host.log"
PI05_DEBUG_DUMP_FILE_DEFAULT="/tmp/pi05_llama_tensor.log"
PI05_DEBUG_DUMP_VALUES_DEFAULT="256"
CHAT_TEMPLATE="vicuna"
NGL="37"
OUTPUT_NAME=""
EXTRA_ARGS=()

usage() {
    cat <<'EOF'
Usage:
  ./scripts/run_pi0.sh [options] cli [-- extra args...]
  ./scripts/run_pi0.sh [options] server [-- extra args...]

Modes:
  cli                     Run llama-mtmd-cli
  server                  Run llama-server

Options:
  -n                      Run under nsys profile
  -d <cuda_device>        CUDA_VISIBLE_DEVICES value (default: 1)
  -o <output_name>        nsys output basename without extension
  --host <host>           Server host (default: 127.0.0.1)
  --port <port>           Server port (default: 8080)
  -h, --help              Show this help

Examples:
  ./scripts/run_pi0.sh cli
  ./scripts/run_pi0.sh -n cli
  ./scripts/run_pi0.sh server
  ./scripts/run_pi0.sh -n -o pi0_server_trace server
  ./scripts/run_pi0.sh -n cli -- --temp 0.2
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        cli|server)
            MODE="$1"
            shift
            break
            ;;
        -n)
            USE_NSYS=1
            shift
            ;;
        -d)
            [[ $# -ge 2 ]] || { echo "Missing value for -d" >&2; exit 1; }
            CUDA_DEVICE="$2"
            shift 2
            ;;
        -o)
            [[ $# -ge 2 ]] || { echo "Missing value for -o" >&2; exit 1; }
            OUTPUT_NAME="$2"
            shift 2
            ;;
        --host)
            [[ $# -ge 2 ]] || { echo "Missing value for --host" >&2; exit 1; }
            HOST="$2"
            shift 2
            ;;
        --port)
            [[ $# -ge 2 ]] || { echo "Missing value for --port" >&2; exit 1; }
            PORT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option before mode: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ -z "${MODE}" ]]; then
    echo "You must specify a mode: cli or server" >&2
    usage >&2
    exit 1
fi

if [[ $# -gt 0 ]]; then
    if [[ "$1" == "--" ]]; then
        shift
    fi
    EXTRA_ARGS=("$@")
fi

cd "${PROJECT_DIR}"

CUDA_HOME="${CUDA_HOME:-}"
if [[ -z "${CUDA_HOME}" ]]; then
    for cuda_dir in "${CUDA_HOME_DEFAULT}" "/usr/local/cuda" "/usr/local/cuda-12.2"; do
        if [[ -f "${cuda_dir}/lib64/libcudart.so.12" ]]; then
            CUDA_HOME="${cuda_dir}"
            break
        fi
    done
fi

if [[ -z "${CUDA_HOME}" || ! -d "${CUDA_HOME}/lib64" ]]; then
    echo "CUDA runtime lib64 not found. Set CUDA_HOME to a CUDA 12 installation." >&2
    exit 1
fi

export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${PROJECT_DIR}/build/bin:${LD_LIBRARY_PATH:-}"
export PI0_ACTION_NOISE_BIN="${PI0_ACTION_NOISE_BIN:-${PI0_ACTION_NOISE_BIN_DEFAULT}}"
export PI0_NORM_STATS_JSON="${PI0_NORM_STATS_JSON:-${PI0_NORM_STATS_JSON_DEFAULT}}"
export PI05_DEBUG_HOST_LOG_FILE="${PI05_DEBUG_HOST_LOG_FILE:-${PI05_DEBUG_HOST_LOG_FILE_DEFAULT}}"
export PI05_DEBUG_DUMP_FILE="${PI05_DEBUG_DUMP_FILE:-${PI05_DEBUG_DUMP_FILE_DEFAULT}}"
export PI05_DEBUG_DUMP_VALUES="${PI05_DEBUG_DUMP_VALUES:-${PI05_DEBUG_DUMP_VALUES_DEFAULT}}"

if [[ ! -f "${PI0_ACTION_NOISE_BIN}" ]]; then
    echo "PI0 action noise file not found: ${PI0_ACTION_NOISE_BIN}" >&2
    exit 1
fi

if [[ ! -f "${PI0_NORM_STATS_JSON}" ]]; then
    echo "PI0 norm stats JSON not found: ${PI0_NORM_STATS_JSON}" >&2
    exit 1
fi

if [[ "${MODE}" == "cli" ]]; then
    BIN="./build/bin/llama-mtmd-cli"
    CMD=(
        "${BIN}"
        -m "${MODEL_PATH}"
        --mmproj "${MMPROJ_PATH}"
        -ngl "${NGL}"
        --chat-template "${CHAT_TEMPLATE}"
        --flash-attn on
    )
else
    BIN="./build/bin/llama-server"
    CMD=(
        "${BIN}"
        -m "${SERVER_MODEL_PATH}"
        --mmproj "${MMPROJ_PATH}"
        -ngl "${NGL}"
        --chat-template "${CHAT_TEMPLATE}"
        --host "${HOST}"
        --port "${PORT}"
        --flash-attn on
    )
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    CMD+=("${EXTRA_ARGS[@]}")
fi

if [[ ! -x "${BIN}" ]]; then
    echo "Binary not found or not executable: ${BIN}" >&2
    exit 1
fi

if [[ "${USE_NSYS}" -eq 1 ]]; then
    if [[ ! -x "${NSYS_BIN}" ]]; then
        echo "nsys binary not found or not executable: ${NSYS_BIN}" >&2
        exit 1
    fi

    mkdir -p "${NSYS_DIR}"

    if [[ -z "${OUTPUT_NAME}" ]]; then
        OUTPUT_NAME="${MODE}_$(date +%Y%m%d_%H%M%S)"
    fi

    NSYS_OUTPUT="${NSYS_DIR}/${OUTPUT_NAME}"

    echo "[run_pi0] mode=${MODE}"
    echo "[run_pi0] CUDA_VISIBLE_DEVICES=${CUDA_DEVICE}"
    echo "[run_pi0] CUDA_HOME=${CUDA_HOME}"
    echo "[run_pi0] nsys output=${NSYS_OUTPUT}.nsys-rep"
    echo "[run_pi0] nsys cmd=${NSYS_BIN} profile --trace=cuda,nvtx,osrt --cpuctxsw=process-tree --sample=none -o ${NSYS_OUTPUT} ${CMD[*]}"

    CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
    "${NSYS_BIN}" profile \
        --trace=cuda,nvtx,osrt \
        --cpuctxsw=process-tree \
        --sample=none \
        -o "${NSYS_OUTPUT}" \
        "${CMD[@]}"
else
    echo "[run_pi0] mode=${MODE}"
    echo "[run_pi0] CUDA_VISIBLE_DEVICES=${CUDA_DEVICE}"
    echo "[run_pi0] CUDA_HOME=${CUDA_HOME}"

    CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${CMD[@]}"
fi
