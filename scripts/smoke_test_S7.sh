#!/usr/bin/env bash
# =============================================================================
# Paper 5 — smoke test for S7 (glacier→FAST routing) on Rhone @ Gletsch (2268)
# =============================================================================
# Verifies the new split_to_fast implementation:
#   1. preprocess succeeds with no Raven schema errors
#   2. the generated .rvi contains the expected
#         :Split RAVEN_DEFAULT PONDED_WATER SURFACE_WATER FAST_RESERVOIR <GlacROF>
#      line (not SLOW_RESERVOIR — that would mean the S7 branch didn't fire)
#   3. Raven runs end-to-end for a few SCEUA iterations
#
# Single config (S7 only), single objective (Q), 20 SCEUA iterations. Expected
# wall clock: ~5-10 minutes on Rhone (~39 km², 20-yr sim).
#
# Run from /home/jberg/Raven-world. Requires Raven-Switzerland conda env.
# =============================================================================

set -euo pipefail

cd /home/jberg/Raven-world
source ~/miniconda3/etc/profile.d/conda.sh
conda activate Raven-Switzerland

SMOKE_NML=/tmp/smoke_2268_S7.yaml
LOG=/home/jberg/Raven-world/logs/smoke_2268_S7_$(date +%Y%m%d_%H%M%S).log
mkdir -p "$(dirname "$LOG")"

# ── 1. Write a minimal single-config smoke namelist ──────────────────────────
cat > "$SMOKE_NML" <<'EOF'
# Auto-generated smoke namelist for S7 verification (Rhone @ Gletsch, 2268).
catchment: '2268'
display_name: 'Rhone @ Gletsch (S7 smoke)'
region: 'switzerland'

start_date: '2000-01-01'
end_date: '2020-12-31'
cali_end_date: '2010-12-31'
warm_up_date: '1990-01-01'

warmup:
  method: 'real'

precip_correction: true

models:
  - SPHY

configurations:
  - glogem_subdaily_opt1_glaciergw_fast        # S7 only

calibration:
  metrics: ['S7_smoke']
  algorithm: SCEUA
  iterations: 20
  ngs: 3

  objectives: [Q]
  Q:
    metric: KGE_NP

future:
  enabled: false
EOF

echo "── Smoke namelist written to $SMOKE_NML"
echo "── Log will stream to $LOG"
echo ""

# ── 2. Run pipeline (preprocess + calibration; no future) ────────────────────
python run_full_pipeline.py "$SMOKE_NML" --skip-future --skip-download 2>&1 | tee "$LOG"

# ── 3. Inspect the generated .rvi for the expected :Split line ───────────────
RVI=$(find /home/jberg/OneDrive/Raven_worldwide/model_runs/catchment_2268/configs/glogem_subdaily_opt1_glaciergw_fast -name "*.rvi" 2>/dev/null | head -1)

if [[ -z "$RVI" ]]; then
    echo ""
    echo "❌ Could not locate generated .rvi for S7. Preprocess may have failed."
    exit 1
fi

echo ""
echo "── Generated .rvi: $RVI"
echo "── Inspecting :Split block:"
grep -A 1 "^   :Split" "$RVI" || echo "   (no :Split line found — S7 did NOT trigger the split_to_fast branch)"

echo ""
if grep -q "PONDED_WATER    SURFACE_WATER   FAST_RESERVOIR" "$RVI"; then
    echo "✅ PASS — :Split line targets FAST_RESERVOIR (S7 wired correctly)"
elif grep -q "PONDED_WATER    SURFACE_WATER   SLOW_RESERVOIR" "$RVI"; then
    echo "❌ FAIL — :Split line still targets SLOW_RESERVOIR (S7 fell through to S2 path)"
    exit 1
else
    echo "❌ FAIL — no :Split line at all (glacier_routing branch did not fire)"
    exit 1
fi

echo ""
echo "── Smoke test complete. Calibration results:"
find /home/jberg/OneDrive/Raven_worldwide/model_runs/catchment_2268/configs/glogem_subdaily_opt1_glaciergw_fast -name "calibration_results_*.csv" -newer "$SMOKE_NML" | head -1
