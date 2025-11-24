import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
import xgboost as xgb
from sklearn.metrics import r2_score
import shap
import matplotlib.pyplot as plt

df = pd.read_excel(file_path)

X = df.drop(target, axis=1)
y = df[target_name]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=seed)

kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
for tr_idx, va_idx in kf.split(X_train):
    model = xgb.XGBRegressor(**params)
    model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
    y_pred = model.predict(X_train.iloc[va_idx])
    _ = r2_score(y_train.iloc[va_idx], y_pred)

final_model = xgb.XGBRegressor(**params)
final_model.fit(X_train, y_train)

explainer = shap.Explainer(final_model)
shap_values = explainer(X_test[features])

fig, ax = plt.subplots()
ax.scatter(X_test[feature], shap_values[:, idx].values)
plt.savefig(output_path)
plt.show()
