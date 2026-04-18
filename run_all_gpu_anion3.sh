#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRIPT_DIR}}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
STM_SCRIPT="${PROJECT_ROOT}/stm_bardeen.py"
PLOT_SCRIPT="${PROJECT_ROOT}/plot_stm.py"
WORK_DIR="${WORK_DIR:-/home/zhangchi/fchk_for_test/anion_3}"
ROW_BATCH="${STM_ROW_BATCH:-48}"
BACKEND="${STM_BACKEND:-gpu}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "[ERROR] Python interpreter not found or not executable: ${PYTHON_BIN}" >&2
    exit 1
fi

cd "${WORK_DIR}"

SUMMARY_FILE="${WORK_DIR}/gpu_batch_summary.tsv"
printf "input\tscan_seconds\ttotal_seconds\tmatrix\toutput_m\toutput_png\n" > "${SUMMARY_FILE}"

shopt -s nullglob
for finp in *.finp; do
    base="${finp%.finp}"
    tmp_finp="$(mktemp "${WORK_DIR}/${base}.gpu.XXXXXX.finp")"
    cp "${finp}" "${tmp_finp}"
    {
        printf "\n"
        printf "STM.Backend %s\n" "${BACKEND}"
        printf "STM.RowBatch %s\n" "${ROW_BATCH}"
    } >> "${tmp_finp}"

    out_m="${WORK_DIR}/${base}_gpu.m"
    out_log="${WORK_DIR}/${base}_gpu.log"
    out_png="${WORK_DIR}/${base}_gpu.png"

    echo "[RUN] ${finp} backend=${BACKEND} row_batch=${ROW_BATCH} python=${PYTHON_BIN}"
    "${PYTHON_BIN}" "${STM_SCRIPT}" "${tmp_finp}" 1 "${BACKEND}" > "${out_m}" 2> "${out_log}"
    plot_output="$("${PYTHON_BIN}" "${PLOT_SCRIPT}" "${out_m}" "${out_png}")"
    rm -f "${tmp_finp}"

    scan_seconds="$(
        grep -oP '(?<=Scan completed in )[^s]+' "${out_log}" | tail -n 1 | tr -d '[:space:]'
    )"
    total_seconds="$(
        grep -oP '(?<=Total time: )[^s]+' "${out_log}" | tail -n 1 | tr -d '[:space:]'
    )"
    matrix_name="$(printf "%s\n" "${plot_output}" | awk -F': ' '/^Matrix:/ {print $2}' | tail -n 1)"

    printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${finp}" \
        "${scan_seconds}" \
        "${total_seconds}" \
        "${matrix_name}" \
        "${out_m}" \
        "${out_png}" >> "${SUMMARY_FILE}"
done

echo "[DONE] Summary: ${SUMMARY_FILE}"
