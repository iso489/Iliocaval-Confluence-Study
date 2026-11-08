# Source metadata (not distributed)

`Code/phase1_characterize/summary/summary_tables.py` reads three metadata tables from the source
datasets. They are not distributed here. Download them from their providers and save them in this
folder under these names.

| File | Source | SHA-256 of the copy used for the paper |
|---|---|---|
| `AMOS.csv` | AMOS release metadata (Zenodo, doi:10.5281/zenodo.7262581); save the download as `raw/AMOS_upstream.csv` and run `python METADATA/normalize_amos_metadata.py`, which corrects one duplicated identifier (the MRI record 500 becomes 600) | `23261233ff390d9f7d3ef08b23659a29cdd8eec4db90e06287d2a6dcd265c7eb` |
| `SAROS.csv` | SAROS release table (The Cancer Imaging Archive, doi:10.25737/SZ96-ZG60) | `66d932459fbb9c294fcd1322ae55b7609d302909d342658de4f641acfe96a92e` |
| `autoPET.csv` | FDG-PET-CT-Lesions clinical metadata (The Cancer Imaging Archive, doi:10.7937/gkr0-xv29) | `0c25094925c1cc0596b9449cc2fe66968eaf766e548b8e18707b667a2c3b38cc` |

A fresh download may differ in bytes (for example, added download columns). The tables reproduce as
long as the columns the analysis reads are the same; the run record then shows a different input
digest.
