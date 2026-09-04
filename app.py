import streamlit as st
import pandas as pd
import pickle

# --------------------------------------------------
# Load trained model
# --------------------------------------------------

with open("employee_attrition_model.pkl", "rb") as file:
    model = pickle.load(file)

st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="👨‍💼",
    layout="centered"
)

st.title("👨‍💼 Employee Attrition Prediction System")
st.write("Enter the employee details below to predict attrition.")

# --------------------------------------------------
# Get columns used during model training
# --------------------------------------------------

expected_columns = list(model.feature_names_in_)

# Get preprocessing step
preprocessor = model.named_steps["preprocessor"]

# Find numerical and categorical columns
numerical_columns = []
categorical_columns = []
categorical_values = {}

for name, transformer, columns in preprocessor.transformers_:

    if name == "num":
        numerical_columns = list(columns)

    elif name == "cat":
        categorical_columns = list(columns)

        # Get OneHotEncoder
        encoder = transformer.named_steps["onehot"]

        for column, categories in zip(
            categorical_columns,
            encoder.categories_
        ):
            categorical_values[column] = list(categories)


# --------------------------------------------------
# Create input form
# --------------------------------------------------

input_data = {}

st.subheader("Employee Information")

for column in expected_columns:

    # Categorical column
    if column in categorical_columns:

        options = categorical_values[column]

        input_data[column] = st.selectbox(
            column,
            options
        )

    # Numerical column
    elif column in numerical_columns:

        input_data[column] = st.number_input(
            column,
            value=0.0
        )

# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("🔮 Predict Attrition"):

    # Create DataFrame
    input_df = pd.DataFrame([input_data])

    # Make sure columns are in exactly the same order
    input_df = input_df[expected_columns]

    # Display input data
    st.subheader("Input Data")
    st.dataframe(input_df)

    # Prediction
    prediction = model.predict(input_df)[0]

    # Probability
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Employee is likely to LEAVE")
    else:
        st.success("✅ Employee is likely to STAY")

    st.write(
        f"**Attrition Probability:** {probability * 100:.2f}%"
    )