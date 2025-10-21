import os
import re
import pandas as pd
import numpy as np

excel_path = os.path.join(os.getcwd(), 'teste banco de dados.xlsx')
print('excel exists:', os.path.exists(excel_path))

candidate = None
for skip in range(0,6):
    tmp = pd.read_excel(excel_path, header=skip)
    non_null_ratio = tmp.notna().sum(axis=1).iloc[0] / tmp.shape[1] if tmp.shape[0] > 0 else 0
    print(f'skip={skip}, shape={tmp.shape}, first-row non-null ratio={non_null_ratio:.3f}')
    if tmp.shape[0] > 0 and non_null_ratio >= 0.2:
        candidate = skip
        break
print('chosen candidate:', candidate)

df = pd.read_excel(excel_path, header=candidate)
raw = pd.read_excel(excel_path, header=None)
print('df.shape:', df.shape)
print('raw.shape:', raw.shape)
print('df.columns:', list(df.columns))

# clean names same as script
def clean_name(s):
    s = str(s).strip()
    s = s.replace('\n', ' ').replace('  ', ' ').strip()
    s = re.sub(r'["\']', '', s)
    s = re.sub(r'\s+', ' ', s)
    s = s.replace(' ', '_')
    return s

new_cols = []
for i, col in enumerate(df.columns):
    col_str = str(col)
    if 'Unnamed' in col_str or col_str.strip() == '':
        if candidate - 1 >= 0 and candidate - 1 < raw.shape[0]:
            candidate_label = raw.iloc[candidate-1, i]
            new_name = clean_name(candidate_label) if pd.notna(candidate_label) and str(candidate_label).strip()!='' else f'feature_{i}'
        else:
            new_name = f'feature_{i}'
    else:
        new_name = clean_name(col_str)
    new_cols.append(new_name)
print('new_cols:', new_cols)

# convert to numeric
for c in df.columns:
    df[c] = pd.to_numeric(df[c].astype(str).str.replace(',','.'), errors='coerce')

# detect target
cands = [c for c in df.columns if re.search(r'(?i)f\s*r\s*3|fr3|fR3|resist', c)]
print('regex candidates:', cands)
if cands:
    target = cands[0]
else:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    target = numeric_cols[-1] if numeric_cols else None
print('target:', target)
if target:
    print('target notna count:', df[target].notna().sum())

for c in df.columns:
    print(c, 'non-null:', df[c].notna().sum())

print('\nSample rows:')
print(df.head(10))
