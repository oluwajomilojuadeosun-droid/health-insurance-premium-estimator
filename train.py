import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

insurance = pd.read_csv(r"C:\Users\aadeo\Downloads\insurance.csv")

numerical_features = ['age', 'bmi', 'children']

categorical_features = ['sex', 'smoker', 'region']

target = ['charges']

X = insurance[categorical_features + numerical_features]
y = insurance[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), numerical_features),
    ]
)

model_pipeline = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42)),
    ]
)

model_pipeline.fit(X_train, y_train)

y_pred = model_pipeline.predict(X_test)

print('\n--- MODEL RESULTS ---')
print(f'R² Accuracy Score: {r2_score(y_test, y_pred) * 100:.1f}%')
print(f'Average Error (MAE): ${mean_absolute_error(y_test, y_pred):,.2f}')

joblib.dump(model_pipeline, 'insurance_model.pkl')
print('\nSuccess! Saved trained pipeline as "insurance_model.pkl"')