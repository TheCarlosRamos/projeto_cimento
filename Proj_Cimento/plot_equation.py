import os
import re
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# localizar excel
excel_path = os.path.join(os.getcwd(), 'teste banco de dados.xlsx')
if not os.path.exists(excel_path):
    raise FileNotFoundError(excel_path)

# detectar header
candidate = None
for skip in range(0,6):
    tmp = pd.read_excel(excel_path, header=skip)
    non_null_ratio = tmp.notna().sum(axis=1).iloc[0] / tmp.shape[1] if tmp.shape[0] > 0 else 0
    if tmp.shape[0] > 0 and non_null_ratio >= 0.2:
        candidate = skip
        break
if candidate is None:
    df = pd.read_excel(excel_path, header=0)
else:
    df = pd.read_excel(excel_path, header=candidate)

# normalizar colunas
df.columns = [str(c).strip().replace('\n',' ').replace('  ',' ').replace(' ','_') for c in df.columns]

# detectar target
candidates = [c for c in df.columns if re.search(r'(?i)f\s*r\s*3|fr3|fR3|resist', c)]
if candidates:
    target_col = candidates[0]
else:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        target_col = numeric_cols[-1]
    else:
        raise ValueError('Não foi possível detectar a coluna alvo.')

# converter para numérico
def to_numeric_series(s):
    return pd.to_numeric(s.astype(str).str.replace(',','.' )
                         .str.replace('[^0-9\.-]','', regex=True), errors='coerce')

df_num = df.copy()
for c in df.columns:
    df_num[c] = to_numeric_series(df[c])

# remover linhas sem target
df_num = df_num.dropna(subset=[target_col])
feature_cols = [c for c in df_num.columns if c != target_col and df_num[c].notna().sum() > 0]

X = df_num[feature_cols].fillna(df_num[feature_cols].median())
y = df_num[target_col]

# carregar modelos e scaler
scaler = joblib.load('scaler_X.joblib')
lr = joblib.load('model_lr.joblib')
rf = joblib.load('model_rf.joblib')

means = scaler.mean_
scales = scaler.scale_
coef_scaled = lr.coef_
intercept = lr.intercept_
adjusted_coefs = coef_scaled / scales
adjusted_intercept = intercept - (coef_scaled * (means / scales)).sum()

# main feature
main_idx = int(np.argmax(np.abs(adjusted_coefs))) if len(adjusted_coefs)>0 else 0
main_feat = feature_cols[main_idx]
print('Main feature:', main_feat)

# grid
grid = np.linspace(X[main_feat].min(), X[main_feat].max(), 300)
grid_X = np.tile(np.median(X.values, axis=0), (grid.shape[0],1))
col_index = list(X.columns).index(main_feat)
grid_X[:, col_index] = grid

# y_eq and y_rf
y_eq = adjusted_intercept + (grid_X * adjusted_coefs).sum(axis=1)
try:
    y_rf = rf.predict(scaler.transform(pd.DataFrame(grid_X, columns=X.columns)))
except Exception:
    y_rf = None

plt.figure(figsize=(9,5))
plt.plot(grid, y_eq, label='Equação (Ridge)', color='C0')
if y_rf is not None:
    plt.plot(grid, y_rf, label='RandomForest', color='C1', alpha=0.8)
plt.scatter(X[main_feat], y, s=12, alpha=0.4, label='Dados (todos)')
plt.xlabel(str(main_feat).replace('_',' '))
plt.ylabel('y (target)')
plt.title('Equação estimada vs RandomForest')
plt.legend()

# montar equação text
eq_lines = ['y = {:.6g}'.format(adjusted_intercept)]
for i, f in enumerate(feature_cols):
    eq_lines.append('{:+.6g} * {}'.format(adjusted_coefs[i], f.replace('_',' ')))
eq_text = '\n'.join(eq_lines)
plt.gcf().text(0.02, 0.02, eq_text, fontsize=8, family='monospace', bbox=dict(facecolor='white', alpha=0.9))

outp = 'equation_plot.png'
plt.tight_layout()
plt.savefig(outp, dpi=150)
print(outp, 'saved, size=', os.path.getsize(outp))
