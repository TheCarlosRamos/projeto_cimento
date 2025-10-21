import os
import re
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

cwd = os.getcwd()
excel_path = os.path.join(cwd, 'teste banco de dados.xlsx')
if not os.path.exists(excel_path):
    raise FileNotFoundError(excel_path)

# detect header
candidate = None
for skip in range(0,6):
    tmp = pd.read_excel(excel_path, header=skip)
    non_null_ratio = tmp.notna().sum(axis=1).iloc[0] / tmp.shape[1] if tmp.shape[0] > 0 else 0
    if tmp.shape[0] > 0 and non_null_ratio >= 0.2:
        candidate = skip
        break
if candidate is None:
    candidate = 0

# read df
df = pd.read_excel(excel_path, header=candidate)
raw = pd.read_excel(excel_path, header=None)

# clean function
import re

def clean_label(x):
    s = str(x).strip()
    s = s.replace('\n',' ').replace('\r',' ')
    s = re.sub(r'\s+', ' ', s)
    s = s.strip()
    return s

new_labels = []
for i, col in enumerate(df.columns):
    col_s = str(col)
    if 'Unnamed' in col_s or col_s.strip() == '' or col_s.startswith('feature_'):
        label = None
        # try line above header
        if candidate-1 >= 0 and candidate-1 < raw.shape[0]:
            cand = raw.iloc[candidate-1, i]
            if pd.notna(cand) and str(cand).strip()!='':
                label = clean_label(cand)
        # try second line above
        if (label is None) and candidate-2 >=0:
            cand = raw.iloc[candidate-2, i]
            if pd.notna(cand) and str(cand).strip()!='':
                label = clean_label(cand)
        if label is None:
            label = f'feature_{i}'
        new_labels.append(label)
    else:
        new_labels.append(clean_label(col_s))

# ensure uniqueness
seen = {}
final_labels = []
for lab in new_labels:
    base = lab
    k = 1
    while lab in seen:
        lab = f"{base}_{k}"
        k += 1
    seen[lab] = True
    final_labels.append(lab)

mapping = dict(zip(list(df.columns), final_labels))
print('Mapping:')
for k,v in mapping.items():
    print(k, '->', v)

# Save mapping for reference
with open('column_mapping.txt','w',encoding='utf-8') as f:
    for k,v in mapping.items():
        f.write(f'{k} -> {v}\n')

# Now rebuild equation and regenerate plots using models
# load models and scaler
scaler = joblib.load('scaler_X.joblib')
lr = joblib.load('model_lr.joblib')
rf = joblib.load('model_rf.joblib')

# prepare numeric df and detect target same as before
for c in df.columns:
    df[c] = pd.to_numeric(df[c].astype(str).str.replace(',','.'), errors='coerce')

candidates = [c for c in df.columns if re.search(r'(?i)f\s*r\s*3|fr3|fR3|resist', c)]
if candidates:
    target_col = candidates[0]
else:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    counts = {c: df[c].notna().sum() for c in numeric_cols}
    target_col = max(counts, key=counts.get)
print('Detected target:', target_col)

# features: try to use the model's training feature names if present
df_num = df.dropna(subset=[target_col])
trained_features = None
if hasattr(lr, 'feature_names_in_'):
    trained_features = list(lr.feature_names_in_)
    print('Using Ridge.feature_names_in_')
elif hasattr(rf, 'feature_names_in_'):
    trained_features = list(rf.feature_names_in_)
    print('Using RF.feature_names_in_ (no Ridge names)')
else:
    print('No feature_names_in_ found; falling back to dataframe-derived features')

if trained_features is not None:
    feature_cols = [c for c in trained_features if c in df_num.columns and c != target_col]
else:
    feature_cols = [c for c in df_num.columns if c != target_col and df_num[c].notna().sum()>0]

print('Feature cols:', feature_cols)

# build X/y using the same order as the trained model
X = df_num[feature_cols].fillna(df_num[feature_cols].median())
y = df_num[target_col]

# compute adjusted coefs for Ridge (align with scaler)
means = scaler.mean_
scales = scaler.scale_
coef_scaled = lr.coef_
intercept = lr.intercept_


# If lengths mismatch between coef_scaled and scaler, attempt to recover the original feature order
if coef_scaled.shape[0] != scales.shape[0] or coef_scaled.shape[0] != len(feature_cols):
    N = coef_scaled.shape[0]
    print(f'Length mismatch: coef={coef_scaled.shape[0]}, scales={scales.shape[0]}, detected_features={len(feature_cols)}')
    # choose top-N numeric columns by non-null count (excluding target)
    numeric_cols = [c for c in df_num.columns if c != target_col and pd.api.types.is_numeric_dtype(df_num[c])]
    counts = {c: df_num[c].notna().sum() for c in numeric_cols}
    sorted_cols = sorted(counts.keys(), key=lambda c: counts[c], reverse=True)
    feature_cols = sorted_cols[:N]
    # keep only those present in df_num
    feature_cols = [c for c in feature_cols if c in df_num.columns]
    # if still short, append other numeric columns
    if len(feature_cols) < N:
        other = [c for c in sorted_cols if c not in feature_cols and c in df_num.columns]
        for c in other:
            feature_cols.append(c)
            if len(feature_cols) >= N:
                break
    print('Recovered feature_cols:', feature_cols)
    # if scaler has more entries than N, subset
    if scales.shape[0] >= N:
        scales = scales[:N]
        means = means[:N]

adjusted_coefs = coef_scaled / scales
adjusted_intercept = intercept - (coef_scaled * (means / scales)).sum()

# build equation text with mapped labels (use original column names mapped to clean labels)
feat_display = [mapping.get(c, str(c)).replace('_',' ') for c in feature_cols]
lines = [f'y = {adjusted_intercept:.6g}']
for i, n in enumerate(feat_display):
    lines.append(f'{adjusted_coefs[i]:+.6g} * {n}')

with open('equation.txt','w',encoding='utf-8') as f:
    f.write('Equação estimada (unidades originais):\n')
    f.write('\n'.join(lines))
print('Wrote equation.txt')

# regenerate coefficients_ridge.png
import matplotlib.pyplot as plt
coef_vals = adjusted_coefs
coef_df = pd.Series(coef_vals, index=feat_display).sort_values()
plt.figure(figsize=(8,6))
coef_df.plot(kind='barh', color='C2')
plt.title('Ridge coefficients (adjusted to original units)')
plt.xlabel('Coefficient')
plt.tight_layout()
plt.savefig('coefficients_ridge.png', dpi=150)
plt.close()
print('Saved coefficients_ridge.png')

# regenerate feature_importances_rf.png
import numpy as np
importances = rf.feature_importances_
idx = np.argsort(importances)[::-1]
plt.figure(figsize=(8,6))
plt.bar([feat_display[i] for i in idx], importances[idx])
plt.xticks(rotation=90)
plt.title('Feature importances (RandomForest)')
plt.tight_layout()
plt.savefig('feature_importances_rf.png', dpi=150)
plt.close()
print('Saved feature_importances_rf.png')

# regenerate equation_plot.png using main feature
main_idx = int(np.argmax(np.abs(adjusted_coefs))) if len(adjusted_coefs)>0 else 0
main_feat = feature_cols[main_idx]
print('Main feature:', main_feat)

# build grid_X using recovered feature_cols order
grid = np.linspace(X[main_feat].min(), X[main_feat].max(), 300)
# compute median_row for feature_cols, ensuring columns exist in X
available_cols = [c for c in feature_cols if c in X.columns]
if len(available_cols) == 0:
    median_row = np.zeros(len(feature_cols))
else:
    median_row = np.median(X[available_cols].values, axis=0)
    # if some feature_cols missing, pad with zeros
    if len(available_cols) < len(feature_cols):
        pad = np.zeros(len(feature_cols) - len(available_cols))
        median_row = np.concatenate([median_row, pad])
grid_X = np.tile(median_row, (grid.shape[0],1))
col_index = feature_cols.index(main_feat)
grid_X[:, col_index] = grid

# ensure adjusted_coefs aligns with feature_cols length
coef_for_calc = adjusted_coefs
if coef_for_calc.shape[0] != len(feature_cols):
    # try to trim or pad with zeros
    N = len(feature_cols)
    if coef_for_calc.shape[0] > N:
        coef_for_calc = coef_for_calc[:N]
    else:
        coef_for_calc = np.concatenate([coef_for_calc, np.zeros(N - coef_for_calc.shape[0])])

y_eq = adjusted_intercept + (grid_X * coef_for_calc).sum(axis=1)
try:
    # prepare df for rf prediction: use feature_cols order and original scaler
    df_grid = pd.DataFrame(grid_X, columns=feature_cols)
    y_rf = rf.predict(scaler.transform(df_grid))
except Exception:
    y_rf = None

plt.figure(figsize=(9,5))
plt.plot(grid, y_eq, label='Equação (Ridge)', color='C0')
if y_rf is not None:
    plt.plot(grid, y_rf, label='RandomForest', color='C1', alpha=0.8)
plt.scatter(X[main_feat], y, s=12, alpha=0.4, label='Dados (todos)')
plt.xlabel(mapping.get(main_feat, main_feat).replace('_',' '))
plt.ylabel('y (target)')
plt.title('Equação estimada vs RandomForest')
plt.legend()

eq_text = '\n'.join(lines)
plt.gcf().text(0.02, 0.02, eq_text, fontsize=8, family='monospace', bbox=dict(facecolor='white', alpha=0.9))
plt.tight_layout()
plt.savefig('equation_plot.png', dpi=150)
plt.close()
print('Saved equation_plot.png')

# report files
files = ['equation.txt','coefficients_ridge.png','feature_importances_rf.png','equation_plot.png']
for f in files:
    print(f, '->', 'EXISTS' if os.path.exists(f) else 'MISSING')
