# Script de treino e relatório (gera modelos e imprime equação)
import os
import re
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV

# localizar excel
excel_path = os.path.join(os.getcwd(), 'teste banco de dados.xlsx')
if not os.path.exists(excel_path):
    raise FileNotFoundError(f'Arquivo não encontrado: {excel_path}')

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
print(f'Loaded Excel with header skip={candidate if candidate is not None else 0}; shape={df.shape}')

# normalizar nomes de colunas
df.columns = [str(c).strip().replace('\n',' ').replace('  ',' ').replace(' ','_') for c in df.columns]
print('Columns:', list(df.columns))

# detectar coluna alvo
candidates = [c for c in df.columns if re.search(r'(?i)f\s*r\s*3|fr3|fR3|resist', c)]
if candidates:
    target_col = candidates[0]
else:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        target_col = numeric_cols[-1]
    else:
        raise ValueError('Não foi possível detectar a coluna alvo.')
print('Target:', target_col)

# converter para numérico
def to_numeric_series(s):
    return pd.to_numeric(s.astype(str).str.replace(',','.')
                         .str.replace('[^0-9\.-]','', regex=True), errors='coerce')

df_num = df.copy()
for c in df.columns:
    df_num[c] = to_numeric_series(df[c])

# remover linhas sem target
df_num = df_num.dropna(subset=[target_col])
feature_cols = [c for c in df_num.columns if c != target_col and df_num[c].notna().sum() > 0]
print('Features used:', feature_cols)

X = df_num[feature_cols].fillna(df_num[feature_cols].median())
y = df_num[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# RandomForest
rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train_s, y_train)
y_pred_rf = rf.predict(X_test_s)
mse_rf = mean_squared_error(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print('RandomForest - MSE:', mse_rf, 'MAE:', mae_rf, 'R2:', r2_rf)

# Ridge
lr = RidgeCV(alphas=(0.1,1.0,10.0))
lr.fit(X_train_s, y_train)
y_pred_lr = lr.predict(X_test_s)
mse_lr = mean_squared_error(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
print('Ridge - MSE:', mse_lr, 'MAE:', mae_lr, 'R2:', r2_lr)

# equação em unidades originais
means = scaler.mean_
scales = scaler.scale_
coef_scaled = lr.coef_
intercept = lr.intercept_
adjusted_coefs = coef_scaled / scales
adjusted_intercept = intercept - (coef_scaled * (means / scales)).sum()
terms = [f"{adjusted_coefs[i]:+.6g} * {feature_cols[i]}" for i in range(len(feature_cols))]
equation = f"y = {adjusted_intercept:.6g} " + ' '.join(terms)
print('\nEstimated equation (original units):')
print(equation)

# salvar
joblib.dump(rf, 'model_rf.joblib')
joblib.dump(lr, 'model_lr.joblib')
joblib.dump(scaler, 'scaler_X.joblib')
print('\nSaved: model_rf.joblib, model_lr.joblib, scaler_X.joblib')
