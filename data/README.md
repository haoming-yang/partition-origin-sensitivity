# Data

Datasets are not included in this repository. Set `DATA_ROOT` to a directory containing the following files:

```text
ETT-small/ETTh1.csv
ETT-small/ETTh2.csv
ETT-small/ETTm1.csv
ETT-small/ETTm2.csv
weather/weather.csv
```

The controlled loader fits the standardizer on training rows only. For the canonical `L=512, H=96` protocol, the ETT hourly row boundaries are `0:8640`, `8640:11520`, and `11520:17420`; the ETT minute row boundaries are `0:34560`, `34560:46080`, and `46080:57600`. Weather uses the split recorded by the loader and the selected configuration.

Obtain datasets from their original public sources and check the corresponding licenses before redistribution.
