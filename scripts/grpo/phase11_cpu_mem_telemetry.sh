#!/usr/bin/env bash

CPU_MEM_PID=""
TRAIN_PID=""

stop_cpu_mem_telemetry() {
  set +e
  if [[ -n "${CPU_MEM_PID:-}" ]] && kill -0 "${CPU_MEM_PID}" >/dev/null 2>&1; then
    kill "${CPU_MEM_PID}" >/dev/null 2>&1 || true
    wait "${CPU_MEM_PID}" >/dev/null 2>&1 || true
  fi
  CPU_MEM_PID=""
  if [[ -f "${CPU_MEM_TELEMETRY_DIR}/process_tree_memory.csv" ]]; then
    "${PYTHON_BIN}" scripts/grpo/summarize_process_tree_memory.py \
      --csv "${CPU_MEM_TELEMETRY_DIR}/process_tree_memory.csv" \
      --output "${CPU_MEM_TELEMETRY_DIR}/process_tree_memory_summary.json" || true
  fi
}

start_cpu_mem_telemetry() {
  local root_pid="${1:-}"
  if [[ -z "${root_pid}" ]]; then
    return 0
  fi
  mkdir -p "${CPU_MEM_TELEMETRY_DIR}"
  "${PYTHON_BIN}" scripts/grpo/sample_process_tree_memory.py \
    --root-pid "${root_pid}" \
    --output "${CPU_MEM_TELEMETRY_DIR}/process_tree_memory.csv" \
    --interval "${CPU_MEM_TELEMETRY_INTERVAL:-5}" \
    --label train \
    > "${CPU_MEM_TELEMETRY_DIR}/process_tree_memory.log" \
    2> "${CPU_MEM_TELEMETRY_DIR}/process_tree_memory.err" &
  CPU_MEM_PID=$!
}

run_phase11_with_cpu_mem_telemetry() {
  "$@" &
  TRAIN_PID=$!
  start_cpu_mem_telemetry "${TRAIN_PID}"
  set +e
  wait "${TRAIN_PID}"
  local train_status=$?
  set -e
  TRAIN_PID=""
  stop_cpu_mem_telemetry
  return "${train_status}"
}
