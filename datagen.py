import pandas as pd

from sklearn.datasets import make_regression

X, y = make_regression(n_samples=1_000_000, n_features=10, n_targets=100)

X_df = pd.DataFrame(data=X)
X_mappings = {name: f'x{idx}' for idx, name in enumerate(X_df.columns.tolist())}
X_df.rename(columns=X_mappings, inplace=True)

y_df = pd.DataFrame(data=y)
y_mappings = {name: f'y{idx}' for idx, name in enumerate(y_df.columns.tolist())}
y_df.rename(columns=y_mappings, inplace=True)

df = pd.concat([X_df, y_df], axis=1).round(3)
df.to_csv('synthetic.csv', index=False)
df.to_parquet('synthetic.parquet', index=False)
