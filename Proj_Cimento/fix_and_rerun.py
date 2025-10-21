"""
fix_and_rerun.py
- Detecta automaticamente o cabeçalho do Excel (mesma heurística usada antes).
- Extrai rótulos das colunas: quando uma coluna aparece como 'Unnamed', tenta usar o valor da linha imediatamente antes do header detectado (se houver) como nome.
- Normaliza nomes (remove quebras, troca espaços por '_').
- Re-treina RandomForest + Ridge, gera métricas, grava modelos e gráficos atualizados, e salva equation.txt e equation_plot.png.
"""
import os
import re
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV

cwd = os.getcwd()
excel_path = os.path.join(cwd, 'teste banco de dados.xlsx')
if not os.path.exists(excel_path):
    raise FileNotFoundError('Excel not found: ' + excel_path)

# detectar header (mesma heurística)
candidate = None
for skip in range(0,6):
    tmp = pd.read_excel(excel_path, header=skip)
    non_null_ratio = tmp.notna().sum(axis=1).iloc[0] / tmp.shape[1] if tmp.shape[0] > 0 else 0
    if tmp.shape[0] > 0 and non_null_ratio >= 0.2:
        candidate = skip
        break
if candidate is None:
    candidate = 0

# ler com header detectado e também sem header para extrair rótulos supracabeçalho
df = pd.read_excel(excel_path, header=candidate)
raw = pd.read_excel(excel_path, header=None)

# função auxiliar para limpar nomes
def clean_name(s):
    s = str(s).strip()
    s = s.replace('\n', ' ').replace('  ', ' ').strip()
    # remove caracteres problemáticos
    s = re.sub(r'["\']', '', s)
    s = re.sub(r'\s+', ' ', s)
    # prefer underscore
    s = s.replace(' ', '_')
    return s

new_cols = []
for i, col in enumerate(df.columns):
    col_str = str(col)
    if 'Unnamed' in col_str or col_str.strip() == '':
        # tentar obter da linha anterior do raw (se existe)
        if candidate - 1 >= 0 and candidate - 1 < raw.shape[0]:
            candidate_label = raw.iloc[candidate-1, i]
            if pd.notna(candidate_label) and str(candidate_label).strip() != '':
                new_name = clean_name(candidate_label)
            else:
                new_name = f'feature_{i}'
        else:
            new_name = f'feature_{i}'
    else:
        new_name = clean_name(col_str)
    # garantir unicidade
    base = new_name
    k = 1
    while new_name in new_cols:
        new_name = f"{base}_{k}"
        k += 1
    new_cols.append(new_name)

# aplicar novos nomes
old_cols = list(df.columns)
df.columns = new_cols
print('Column rename map:')
for o,n in zip(old_cols, new_cols):
    if o != n:
        print(f"  {o} -> {n}")

# converter colunas para numérico
def to_numeric_series(s):
    return pd.to_numeric(s.astype(str).str.replace(',', '.').str.replace('[^0-9\.-]', '', regex=True), errors='coerce')

for c in df.columns:
    df[c] = to_numeric_series(df[c])
# detectar target
candidates = [c for c in df.columns if re.search(r'(?i)f\s*r\s*3|fr3|fR3|resist', c)]
if candidates:
    target_col = candidates[0]
else:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        # choose the numeric column with the most non-null values as fallback target
        counts = {c: df[c].notna().sum() for c in numeric_cols}
        target_col = max(counts, key=counts.get)
    else:
        raise ValueError('Não foi possível detectar a coluna alvo.')
print('Target column chosen:', target_col)

# preparar X,y
df_num = df.dropna(subset=[target_col])
# Primeiro, usar todas as colunas exceto o target
feature_cols = [c for c in df_num.columns if c != target_col]
# Remover colunas sem nenhum valor válido
feature_cols = [c for c in feature_cols if df_num[c].notna().sum() > 0]

# Fallback: se ainda estiver vazio, pegar colunas numéricas do dataframe original
if not feature_cols:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c != target_col]

if not feature_cols:
    raise ValueError('Nenhuma feature disponível após pré-processamento. Verifique os dados do Excel.')

print('Features used:', feature_cols)

X = df_num[feature_cols].fillna(df_num[feature_cols].median())
y = df_num[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# RF
rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train_s, y_train)
y_pred_rf = rf.predict(X_test_s)

# Ridge
lr = RidgeCV(alphas=(0.1,1.0,10.0))
lr.fit(X_train_s, y_train)
y_pred_lr = lr.predict(X_test_s)

# métricas
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
print('RandomForest R2:', r2_score(y_test, y_pred_rf))
print('Ridge R2:', r2_score(y_test, y_pred_lr))

# ajustar coeficientes para unidades originais
means = scaler.mean_
scales = scaler.scale_
coef_scaled = lr.coef_
intercept = lr.intercept_
adjusted_coefs = coef_scaled / scales
adjusted_intercept = intercept - (coef_scaled * (means / scales)).sum()

# salvar modelos e scaler
joblib.dump(rf, 'model_rf.joblib')
joblib.dump(lr, 'model_lr.joblib')
joblib.dump(scaler, 'scaler_X.joblib')
print('Saved model_rf.joblib, model_lr.joblib, scaler_X.joblib')

# salvar equation.txt (formatado)
feat_display = [c.replace('_',' ') for c in feature_cols]
with open('equation.txt','w',encoding='utf-8') as f:
    f.write('Equação estimada (unidades originais):\n')
    f.write(f'y = {adjusted_intercept:.6g}\n')
    for i, n in enumerate(feat_display):
        f.write(f"{adjusted_coefs[i]:+.6g} * {n}\n")
print('Saved equation.txt')

# Plots: learning curve, predicted vs actual, residuals, coefficients, feature importances, equation_plot
import seaborn as sns
sns.set(style='whitegrid')

# learning curve (RF)
plt.figure(figsize=(8,6))
train_sizes, train_scores, test_scores = learning_curve(rf, X, y, cv=5, scoring='r2', n_jobs=-1, train_sizes=np.linspace(0.1,1.0,5))
plt.plot(train_sizes, np.mean(train_scores,axis=1), label='Train R2')
plt.plot(train_sizes, np.mean(test_scores,axis=1), label='CV R2')
plt.xlabel('Training examples')
plt.ylabel('R2')
plt.legend()
plt.tight_layout()
plt.savefig('learning_curve_rf.png', dpi=150)
plt.close()

# predicted vs actual
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_rf, alpha=0.6)
mn = min(min(y_test), min(y_pred_rf))
mx = max(max(y_test), max(y_pred_rf))
plt.plot([mn,mx],[mn,mx],'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted (RF)')
plt.tight_layout()
plt.savefig('predicted_vs_actual_rf.png', dpi=150)
plt.close()

# residuals
residuals = y_test - y_pred_rf
plt.figure(figsize=(6,4))
sns.histplot(residuals, kde=True)
plt.tight_layout()
plt.savefig('residuals_rf.png', dpi=150)
plt.close()

# coefficients ridge
coef_vals = adjusted_coefs
coef_df = pd.Series(coef_vals, index=feat_display).sort_values()
plt.figure(figsize=(8,6))
coef_df.plot(kind='barh')
plt.tight_layout()
plt.savefig('coefficients_ridge.png', dpi=150)
plt.close()

# feature importances
importances = rf.feature_importances_
idx = np.argsort(importances)[::-1]
plt.figure(figsize=(8,6))
plt.bar([feat_display[i] for i in idx], importances[idx])
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('feature_importances_rf.png', dpi=150)
plt.close()

# equation plot: choose main feature
main_idx = int(np.argmax(np.abs(adjusted_coefs))) if len(adjusted_coefs)>0 else 0
main_feat = feature_cols[main_idx]
print('Main feature for plotting:', main_feat)

grid = np.linspace(X[main_feat].min(), X[main_feat].max(), 300)
grid_X = np.tile(np.median(X.values, axis=0), (grid.shape[0],1))
col_index = list(X.columns).index(main_feat)
grid_X[:, col_index] = grid

# predictions
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
plt.xlabel(main_feat.replace('_',' '))
plt.ylabel('y (target)')
plt.title('Equação estimada vs RandomForest')
plt.legend()

# equation text
eq_lines = [f'y = {adjusted_intercept:.6g}']
for i, f in enumerate(feat_display):
    eq_lines.append(f'{adjusted_coefs[i]:+.6g} * {f}')
eq_text = '\n'.join(eq_lines)
plt.gcf().text(0.02, 0.02, eq_text, fontsize=8, family='monospace', bbox=dict(facecolor='white', alpha=0.9))
plt.tight_layout()
plt.savefig('equation_plot.png', dpi=150)
plt.close()
print('Saved equation_plot.png')

# listar arquivos gerados
files = ['learning_curve_rf.png','predicted_vs_actual_rf.png','residuals_rf.png','coefficients_ridge.png','feature_importances_rf.png','equation.txt','equation_plot.png','model_rf.joblib','model_lr.joblib','scaler_X.joblib']
for f in files:
    print(f, '->', 'EXISTS' if os.path.exists(f) else 'MISSING')
