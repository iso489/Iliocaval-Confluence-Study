# Iliocaval label disconnection in TotalSegmentator CT segmentation

Analysis code and derived data for the manuscript "Iliocaval Label Disconnection in TotalSegmentator
CT Segmentation and Its Dependence on the Sampling Grid: A Multicohort Retrospective Study".

TotalSegmentator 2.17.0 labels the inferior vena cava (IVC) and both common iliac veins. For 1,873
measurable CT examinations from AMOS (493), SAROS (366) and autoPET (1,014), the code tests whether
the largest IVC and left iliac components are joined by a path through venous voxels, on the native CT
grid and on the network's 1.5-mm inference grid, and audits the 600 measurable released
TotalSegmentator annotations the same way.

## What is here

| Path | Contents |
|---|---|
| `Code/phase0_data/` | The case rosters, the autoPET identifier crosswalk, and the export of the venous and vertebral masks from TotalSegmentator's output |
| `Code/phase1_characterize/` | The measurement, statistics and model-grid code, with its tests |
| `Code/phase1_characterize/summary/` | The code that builds the summary tables behind the reported numbers |
| `Code/common/` | Shared configuration, statistics and file helpers |
| `Results/phase1_characterize/` | The per-examination tables derived from the TotalSegmentator 2.17.0 masks, keyed to the public case identifiers, and the summary tables (`summary/t1_*.csv`, described in `summary/RESULTS_SUMMARY.md`) |
| `METADATA/` | Where to put the three source metadata tables, which are not distributed (`METADATA/README.md`) |
| `environment.yml` | The analysis environment |

No CT images, segmentation masks or source metadata files are distributed. The per-examination
tables carry only the source fields the analysis used: the autoPET acquisition dates, which the public
case identifiers contain, and the released annotations' age and pathology labels.

## Setup

```bash
conda env create -f environment.yml
conda activate ivcs-analysis
```

## Run the tests

```bash
python -m pytest -q Code
```

The tests that read the source metadata tables are skipped until those tables are in `METADATA/`.

## Reproduce the summary tables

With the three metadata tables in `METADATA/`:

```bash
python Code/phase1_characterize/summary/summary_tables.py
```

This rewrites the tables in `Results/phase1_characterize/summary/`, with `RESULTS_SUMMARY.md` and
`run_record.json`, from the per-examination tables; afterwards `git status` should show no change.
`t1_reference_arterial.csv` comes from `reference_arterial.py`, which downloads the released
annotations' arterial masks from Zenodo. The measurement steps need the CT images and the
TotalSegmentator masks; their code is in `Code/phase1_characterize/`.

`Results/phase1_characterize/class_score_resampling/` holds the class-score experiment (Table S5).
`class_score_resampling.py` wrote its per-examination table and needs a GPU, the CT images and
TotalSegmentator 2.17.0; the intervals are computed from that table alone:

```bash
python Code/phase1_characterize/summary/class_score_intervals.py
```

## Data sources

| Dataset | Persistent identifier | License |
|---|---|---|
| AMOS | doi:10.5281/zenodo.7262581 | CC BY 4.0 |
| SAROS | doi:10.25737/SZ96-ZG60 | CC BY 4.0 (the source CT collections keep their own terms) |
| autoPET (FDG-PET-CT-Lesions) | doi:10.7937/gkr0-xv29 | CC BY 4.0, under The Cancer Imaging Archive's access policy |
| TotalSegmentator dataset v2.0.1 | doi:10.5281/zenodo.10047292 | CC BY 4.0 |

## License

Apache License 2.0 (`LICENSE`), for the code and the derived tables. The source datasets keep their
own licenses.
