import pandas as pd
import pyarrow.parquet as pq

table = pq.read_table("data/raw/train_chunk.parquet")
df = table.to_pandas()
print("Unique labels in chunk 0:", df['label'].unique())
print("Value counts:\n", df['label'].value_counts())
