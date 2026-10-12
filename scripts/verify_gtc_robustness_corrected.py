"""Re-runs verify_gtc_robustness.py's checks (mass decomposition, image-grouped CV,
box-geometry confound / geometry-only control, AUPRC) on CORRECTED labels, for the
Full-reframe (Option B) rewrite. Reuses that module's functions unchanged; only the
population's `grounded` labels are swapped to `grounded_corrected` before calling them.

Usage: python verify_gtc_robustness_corrected.py
"""
import verify_gtc_robustness as base


def main():
    r, _, gtc20, align8, mass, groups = base.load_table1_population()
    y = [0 if x["grounded_corrected"] else 1 for x in r]
    for x in r:
        x["grounded"] = x["grounded_corrected"]

    print(f"Loaded {len(r)} mentions ({sum(y)} hallucinated) -- CORRECTED labels\n")
    base.check_mass_decomposition(y, gtc20, mass)
    base.check_image_grouped_cv(y, gtc20, align8, groups)
    base.check_box_geometry_confound(y, gtc20, align8, groups)
    base.check_independent_layer_selection(r)
    base.check_auprc(y, gtc20, align8, groups)
    base.check_error_analysis(r, y, gtc20, align8, groups)
    base.check_geometry_only_control(r)


if __name__ == "__main__":
    main()
