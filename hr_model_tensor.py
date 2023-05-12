import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Create the input function
def input_fn(features, labels, batch_size=52, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(features))
    dataset = dataset.batch(batch_size)
    return dataset


# Define the attributes
numerical_features = ["Age", "MonthlyIncome", "YearsAtCompany",
                      "hrs"]
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

# Read the data
data = pd.read_csv("full_dataset.csv")

# Split the data into features and target
X = data[numerical_features + categorical_features]
y = data[target]

# Convert categorical features to string type
X[categorical_features] = X[categorical_features].astype(str)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert target variable to float type
y_train = y_train.map({"No": 0, "Yes": 1})
y_test = y_test.map({"No": 0, "Yes": 1})

# Scale the numerical features
scaler = StandardScaler()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

# Create feature columns for numeric features
numeric_columns = []
for feature_name in numerical_features:
    numeric_columns.append(tf.feature_column.numeric_column(feature_name))

# Create feature columns for categorical features
categorical_columns = []
for feature_name in categorical_features:
    vocabulary = X_train[feature_name].unique()
    cat_column = tf.feature_column.categorical_column_with_vocabulary_list(
        feature_name, vocabulary
    )
    indicator_column = tf.feature_column.indicator_column(cat_column)
    categorical_columns.append(indicator_column)

# Combine feature columns
feature_columns = numeric_columns + categorical_columns

# Create the TensorFlow model
model = tf.keras.Sequential(
    [
        tf.keras.layers.DenseFeatures(feature_columns),
        tf.keras.layers.Dense(units=1, activation="sigmoid"),
    ]
)

# Compile the model
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=["accuracy"]
)

# Train the model
model.fit(x=input_fn(X_train, y_train), epochs=30)

# Evaluate the model on the test set
eval_result = model.evaluate(x=input_fn(X_test, y_test, shuffle=False))
print("\nTest set accuracy: {:0.3f}\n".format(eval_result[1]))

# Get the model's weights
weights = model.get_weights()[0]

# Display the coefficients
print("Coefficient weights:")
for i, feature in enumerate(numerical_features + categorical_features):
    print(f"{feature:25}: {weights[i][0]:>20}")
