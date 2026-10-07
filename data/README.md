# Data

Datasets are not included in this repository. The default is repository `data/`;
set `DATA_ROOT` to use another directory containing the following files:

```text
ETT-small/ETTh1.csv
ETT-small/ETTh2.csv
ETT-small/ETTm1.csv
ETT-small/ETTm2.csv
weather/weather.csv
```

The controlled loader fits the standardizer on training rows only. For the canonical `L=512, H=96` protocol, the ETT hourly row boundaries are `0:8640`, `8640:11520`, and `11520:17420`; the ETT minute row boundaries are `0:34560`, `34560:46080`, and `46080:57600`. Weather uses `train_end=floor(0.7*N)`, `test_length=floor(0.2*N)` and `validation_end=N-test_length`, with the remaining rows for validation. Targets stay within their split; histories may include preceding rows. Windows advance one time step. ETTh1/ETTh2 counts are 8033/2785/5805; ETTm1/ETTm2 are 33953/11425/11425. The H192 ETTh1 counts are 7937/2689/5709.

The full-split mixer uses `PARTITION_ORIGIN_DATA_ROOT` if set, then `DATA_ROOT`,
then `data/`. It requires the historical 17420-row ETTh1 file and checks the
expected window counts. Do not silently truncate a different file to pass.
Per-run source/dataset hashes record the actual CSV bytes. Dry-run and the
synthetic smoke experiment do not read these CSVs.

Obtain datasets from their original public sources and check the corresponding licenses before redistribution.
