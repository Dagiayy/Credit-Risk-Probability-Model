# Data directory (not version-controlled)

This project uses the Xente transactions dataset (~95k rows: TransactionId,
CustomerId, ProductCategory, ChannelId, ProviderId, Amount, Value,
TransactionStartTime, PricingStrategy, FraudResult, ...).

The dataset is not redistributed in this repository. To run the pipeline
locally, place the raw CSV at:

```
data/raw/data.csv
```

Everything under `data/` is git-ignored (see `.gitignore`) — generated files
(`data/processed/*.csv`, the fitted feature pipeline, etc.) are reproducible
from the raw CSV by running the pipeline described in the root `README.md`,
so they don't need to be committed.
