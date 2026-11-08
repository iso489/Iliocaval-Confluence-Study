"""Version-controlled analysis constants for the IVC-LCIV confluence-gap study.

They are echoed verbatim into the provenance record written next to each output (see
stats.write_provenance)."""

# No published source defines a 2.5-7.5 mm band, a ~5 mm buffer, or any fixed millimetre margin at
# the iliocaval confluence: the compression literature works in PERCENTAGES (>25 / >50 / >61 / >70%
# area or diameter) and at most an absolute-diameter cutoff, none of which is a gap buffer. These
# band edges are an INTERNAL GAP-CHARACTERIZATION CONSTRUCT and nothing more. The emitted column key
# is `band_2p5_7p5_pct`.
PRESENT_MM = 0.5      # < one isotropic voxel (1-3 mm typical CT) -> "no resolvable gap"
BAND_LO_MM = 2.5      # nominal descriptive band, lower edge (internal construct, not clinical)
BAND_HI_MM = 7.5      # nominal descriptive band, upper edge (internal construct, not clinical)
LARGE_GAP_MM = 30.0  # tail/outlier threshold (n_gt_30mm) and the default axis cap of the
                      # stats.py figures; a figure that truncates its axis annotates its own
                      # above-cap count.

# --- Axial vs lateral decomposition (SCALE-INVARIANT) ---------------------
# Classification is on the lateral leg in mm (sqrt(max(0, d^2 - z^2))), NOT on
# the difference |d - z| (which shrinks as the gap grows and so under-detects
# lateral divergence at large gaps). lateral leg is an UPPER BOUND on the true
# lateral separation (the nearest-pair axial component may exceed z_gap).
LATERAL_TOL_MM = 1.0  # lateral leg <= 1 mm   -> gap is essentially axial
AXIAL_TOL_MM = 0.5    # axial extent <= 0.5 mm (sub-voxel) -> gap is lateral
INVARIANT_SLACK_MM = 0.15  # tolerance for the geometric invariant d >= z

# --- Censoring ------------------------------------------------------------
CENSOR_BOUNDARY_MM = 0.0   # producer clamps axial gap with max(0, ...)

# --- Correlate analysis ---------------------------------------------------
MIN_GROUP_N = 10      # min cases/level to test; KW with <10/level is fragile
ALPHA = 0.05

CONFOUNDED_VARIABLES = ("scanner", "region", "seg_quality", "patient_size", "contrast")

# Age bands: clinical convention; the cut points are arbitrary and reported as such.
AGE_BANDS = ((40, "<40"), (60, "40-59"), (75, "60-74"), (None, "75+"))

# --- Reproducibility ------------------------------------------------------
SEED = 20240611
N_BOOT = 2000
# Confidence-interval method for medians and the apex root. "percentile" is kept available for
# comparison.
CI_METHOD = "bca"

# --- QC -------------------------------------------------------------------
QC_MAX_GAP_MM = 200.0   # a 3D gap above this is physiologically implausible -> flag
QC_MIN_VOXELS = 50      # a vessel mask below this voxel count is likely degenerate

# --- Bifurcation apex estimation (measure_bifurcation.py) -----------------
# The wall-contact estimand is implemented in statlib.wall_contact_apex. The equivalent-circle
# 'tangency' contact is the cross-check; it discards lumen shape. sep(z) is fit (not d_min) because
# d_min PLATEAUS near contact (labels stop before the lumens touch) and is a poor extrapolation
# target, whereas centroids keep converging. Its per-case CI is INDICATIVE.
BIF_WINDOW_MM = 20.0    # cranial window over which convergence is ~linear
BIF_MIN_SLICES = 6      # need >=6 shared slices (a 5-point bootstrap is coarse
                        # and anti-conservative under slice autocorrelation)
BIF_SLOPE_TOL = 1e-6    # |slope| <= tol -> not converging, intercept undefined
BIF_WIDE_CI_MM = 10.0   # apex 95% CI half-width above this -> WIDE_CI
BIF_NBOOT = 2000        # bootstrap resamples for the apex CI
BIF_BLOCK_SLICES = 3    # CIRCULAR MOVING-BLOCK bootstrap length (in slices) for the
                        # per-case apex CI. Adjacent CT slices are autocorrelated, so a block
                        # bootstrap is a more coherent resampling model than i.i.d. slices. Circular
                        # starts equalize boundary inclusion; coverage remains uncalibrated and the
                        # CI is indicative. Falls back to i.i.d. when window <= block. Data-driven
                        # block selection (Hall, Horowitz & Jing 1995; Politis & White 2004) makes l
                        # grow with n -- an l ~ n^(1/3) rule gives about 1.8 at n = 6 and 2.7 at n =
                        # 20. A fixed 3 is a CHOICE that departs from those rules at the small end.
                        # Block-bootstrap theory is asymptotic in n with l/n -> 0; at the
                        # BIF_MIN_SLICES = 6 floor, l/n = 0.5. No coverage claim is available from
                        # theory at that size and none is made: every per-case apex CI is reported
                        # as indicative, no apex result is a headline claim in this release, and the
                        # BCa acceleration built on this scheme is a coherent heuristic rather than
                        # a guaranteed one (statlib._jackknife_delete_block). The block length is
                        # fixed, not adaptive; a headline use of the apex would need l re-derived
                        # and coverage shown on a known-apex phantom.

# --- Apex estimand (literature-informed) ---------------------------------- PRIMARY method
# 'wallgap' is SHAPE-AGNOSTIC: it uses the per-slice MINIMUM physical voxel-box surface distance
# between the two common iliac vein masks (no circular-lumen assumption) and extrapolates to d_min =
# 0 (true wall contact = confluence). The apex is an automated PROXY for the confluence, NOT a new
# clinical definition. The wall-contact/surface-tangency concept itself has precedent in the
# VMTK/Antiga bifurcation reference systems (apex = where the two vessel surfaces /
# maximal-inscribed-sphere tubes meet); we adapt it to stacked 2-D segmentation cross-sections with
# extrapolation across the UNLABELED TotalSegmentator gap and bootstrap-CI propagation. The
# literature defines the confluence visually (most caudal single-IVC slice / most cranial two-CIV
# slice) or by vertebral level (typically L5; Moussallem 2012 65.75% at L5 (p. 870) is the
# paper's Discussion WHOLE-L5 figure -- its primary results table reports 29.3% at "upper L5"
# specifically; we cite the broader whole-L5 figure to match this repo's L5-body band [4.5, 5.5).
BIF_APEX_METHOD = "wallgap"   # "wallgap" (shape-agnostic, PRIMARY) | "tangency"
BIF_CURV_ALPHA = 0.05         # quadratic term p < this over the window -> NONLINEAR
BIF_VARIANT_FRAC = 0.50       # >1 connected component on either CIV in at least this share
                              # of window slices -> VARIANT. This is a
                              # mask-fragmentation/topology flag, not a validated anatomical-
                              # variant diagnosis; the two-vein proxy may not apply.
BIF_LIT_ZONE_MM = 15.5        # literature junction-height reference (Pirro N et al.
                              # 2005, Surg Radiol Anat 27:137-141, DOI 10.1007/
                              # s00276-004-0301-8, PMID 15611835) for the
                              # confluence-zone comparison.
                              # Pirro's "height" is an extent of the junction, not the
                              # position of one point; the abstract shows this three ways:
                              # (i) POSITION is reported separately -- "between L4 and S1,
                              # most often at the level of L5 (64%)" -- so a positional "height" would
                              # duplicate it; (ii) the paper measures a WIDTH of the same
                              # object -- "not related to the width of the iliocava
                              # junction" -- and a point has no width, so height is its
                              # second dimension; (iii) "the iliocava junction COVERED THE
                              # WHOLE OF THE L5-S1 DISC in 12% of cases", and a coordinate
                              # cannot cover a disc. Relations to other landmarks are stated
                              # as distances ("mean distance ... to aortic bifurcation 19
                              # mm"), which is what a coordinate-style measure looks like
                              # here. Limit, stated rather than implied: this is the
                              # abstract, not the full text.
                              # NB: do NOT confuse with Akhgar's 30.2 mm, which is a
                              # confluence->sacral-promontory landmark distance, not a
                              # junction-height span.
BIF_VALID_TOL_MM = 17.0       # apex-vs-reference agreement tolerance: THIS PROJECT'S OWN
                              # uncalibrated choice, not a field standard; it gates nothing.

# --- Phase-1 reporting: cohort-flow denominators & analysis knobs ----------- The primary Phase-1
# summary (printed + provenance-stamped, see stats.py).
PRIMARY_INFERENCE = "per-cohort median 3D surface gap with BCa 95% CI"

# Dataset-level eligible-case denominators for transparent cohort-flow accounting. The
# measurable/empty/unparsed table only closes over cases that ENTERED the pipeline;
# these record the upstream denominator + the documented selection so the flow
# 'eligible -> processed -> measurable/empty/unparsed' is auditable (AMOS analyzes
# 493 of the 500 CT including the GE scanner stratum (n=120); autoPET's 1014
# studies come from 900 unique subjects). From the metadata CSVs.
COHORT_ELIGIBLE = {
    # cohort: (label, n_eligible, selection_note)
    "amos":    ("AMOS CT cases (metadata IDs)", 500,
                "493 processed (the six participants under 18 are absent from the TotalSegmentator "
                "2.17.0 export, and amos_0143 is excluded as out of field); 493 measurable, 0 empty; "
                "all four scanner strata represented (GE n=120, Philips 111, "
                "Siemens 126, Toshiba 136)"),
    "saros":   ("SAROS openly-available vein-bearing CT (abdomen+wholebody)", 428,
                "428 openly downloadable of 600 vein-bearing (172 in restricted collections); 392 processed "
                "(of the 36 eligible studies not analysed, 30 abdomen studies were processed under 2.17.0 and "
                "excluded on confluence eligibility, the confluence lying below the field of view as in "
                "amos_0143, and 6 are absent from the 2.17.0 export); 26 empty of 392 (6.6%; 56 of 422 = 13.3% counting the 30 "
                "exclusions), concentrated in one level of the recorded `region` stratifier (26/26 = 100% "
                "abdomen). CONSISTENT WITH MAR GIVEN region; MNAR is NOT identified and no analysis here "
                "tests it. A pelvic-FOV truncation is a plausible hypothesis, not a measured mechanism. The "
                "two-sided worst-case (Manski) bound in _cross_cohort.txt, published on the processed and the "
                "eligible frame, is its only sensitivity"),
    "autopet": ("autoPET-I FDG studies", 1014,
                "1014 studies from 900 unique subjects (81 subjects contribute 2-5 "
                "studies = 114 repeat-patient studies); study-level primary summaries are "
                "descriptive, with an earliest-study-per-subject sensitivity "
                "analysis for independent units"),
}
# The TotalSegmentator version that produced the masks this study measures (the confluence gap is
# its artifact, so the producer version is pinned). The 24,530 masks carry per-file SHA-256 in
# their own MANIFEST and were verified voxel-for-voxel against the source multilabel volume.
TOTALSEG_VERSION = "2.17.0 (nnU-Net 2.8.1, torch 2.13.0; task `total`, full resolution, --ml; " \
                   "per-file SHA-256 in the relevant-mask MANIFEST, 24,530/24,530 verified " \
                   "against the source multilabel volume)"

DATASET_VERSION = {
    "amos":    "AMOS22 v1.0 (from amos22-dataset/dataset.json)",
    "saros":   "SAROS (TCIA) NIfTI-files-v2",
    "autopet": "autoPET-I — TCIA collection 'FDG-PET-CT-Lesions'; 1014 studies / 900 "
               "subjects (see METADATA/autoPET.csv + autopet_cases.txt)",
}

# Present-rate threshold sensitivity sweep (mm): report gap-present at each so the
# headline does not hinge on PRESENT_MM alone (it is robust in practice).
PRESENT_SWEEP_MM = (0.5, 1.0, 2.0)


# Version-controlled and echoed into provenance.
# ===========================================================================

OVERLAP_CONNECTIVITY = 26              # 3-D connectivity for component counts and Betti numbers
