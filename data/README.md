# Data Indexes

This directory contains metadata indexes, not raw images.

- `pilot_index.json`: first 300-image COCO val2014 batch.
- `pilot_index_extra.json`: independent second 300-image COCO batch.
- `openimages_index.json`: 300-image Open Images validation batch.

Each row includes an image identifier, dimensions, object annotations, and a
repository-relative `local_path` populated by the download scripts. COCO and
Open Images remain under their respective licenses.

The CHAIR synonym resource used by the installable package is stored at
`src/larc_audit/data/chair_synonyms.txt`.
