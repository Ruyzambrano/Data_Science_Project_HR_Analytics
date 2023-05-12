import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load the dataframe
df = pd.read_csv("full_dataset.csv")

# Specify the column names and their data types
numerical_features = ["Age", "MonthlyIncome", "YearsAtCompany", "hrs"]
categorical_features = [
    "EnvironmentSatisfaction",
    "BusinessTravel",
    "JobSatisfaction",
    "WorkLifeBalance",
    "Department",
    "Education",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "PerformanceRating",
]
target = "Attrition"

# Split the data into features and target variable
X = df[numerical_features + categorical_features]
y = df[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Preprocessing for numerical features
numerical_transformer = StandardScaler()

# Preprocessing for categorical features
categorical_transformers = []
categorical_feature_names = []
for feature in categorical_features:
    encoder = Pipeline(steps=[("onehot", OneHotEncoder(drop="first"))])
    categorical_transformers.append((feature, encoder, [feature]))
    encoder.fit(X_train[[feature]])  # Fit the encoder
    feature_names = encoder.named_steps["onehot"].get_feature_names_out([feature])
    categorical_feature_names.extend(feature_names)

# Combine the preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        *categorical_transformers,
    ]
)

# Create a pipeline with preprocessing and model
model = Pipeline(
    steps=[("preprocessor", preprocessor), ("classifier", LogisticRegression())]
)

# Fit the pipeline on the training data
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Generate the classification report
report = classification_report(y_test, y_pred)

# Convert the classification report to a DataFrame
report_df = pd.DataFrame(
    classification_report(y_test, y_pred, output_dict=True)
).transpose()

# Print the DataFrame
print("\nModel Statistics:")
print(report_df)

print("\nCoefficients")
coefficients = model.named_steps["classifier"].coef_
feature_names = numerical_features + categorical_feature_names
coefficients_df = pd.DataFrame(
    {"Feature": feature_names, "Coefficient": coefficients[0]}
)
coefficients_df = coefficients_df.sort_values(by="Coefficient", ascending=False)
print(coefficients_df)
