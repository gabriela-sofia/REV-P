# Georeferencing requirements

At least three explicit valid GCPs, a known CRS and a sufficient transform are required. Embedded
georeferencing or a world file would also qualify. A visual image, map envelope or OCR guess alone
does not close TP2. The current candidate uses printed EPSG:32725 grid ticks and normalizes to
EPSG:4326; human review remains mandatory.
