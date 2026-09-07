import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="👨‍💼",
    layout="centered"
)

st.title("👨‍💼 Employee Attrition Prediction System")
st.write("Enter the employee details below to predict attrition.")

# --------------------------------------------------
# Load Trained Model
# --------------------------------------------------

try:
    model = joblib.load("employee_attrition_model.pkl")
    st.success("✅ Model loaded successfully!")

except Exception as e:
    st.error("❌ Model could not be loaded.")
    st.write("Error:", e)
    st.stop()

# --------------------------------------------------
# Get Features Used During Training
# --------------------------------------------------

try:
    expected_columns = list(model.feature_names_in_)

except Exception:
    st.error("❌ Could not find feature names in the model.")
    st.stop()

# --------------------------------------------------
# Get Preprocessor
# --------------------------------------------------

try:
    preprocessor = model.named_steps["preprocessor"]

except Exception:
    st.error("❌ Preprocessor not found in the trained model.")
    st.stop()

# --------------------------------------------------
# Identify Numerical and Categorical Columns
# --------------------------------------------------

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

        # Get possible values for each categorical column
        for column, categories in zip(
            categorical_columns,
            encoder.categories_
        ):
            categorical_values[column] = list(categories)

# --------------------------------------------------
# Employee Information
# --------------------------------------------------

st.subheader("📝 Employee Information")

input_data = {}

for column in expected_columns:

    # ----------------------------------------------
    # Categorical Features
    # ----------------------------------------------

    if column in categorical_columns:

        options = categorical_values[column]

        input_data[column] = st.selectbox(
            f"{column}",
            options
        )

    # ----------------------------------------------
    # Numerical Features
    # ----------------------------------------------

    elif column in numerical_columns:

        input_data[column] = st.number_input(
            f"{column}",
            value=0.0
        )

# --------------------------------------------------
# Prediction
# --------------------------------------------------

st.subheader("🔮 Prediction")

if st.button("Predict Attrition"):

    # Create DataFrame
    input_df = pd.DataFrame([input_data])

    # Keep exactly the same column order
    input_df = input_df[expected_columns]

    # ----------------------------------------------
    # Display Input
    # ----------------------------------------------

    st.subheader("📋 Employee Details")

    st.dataframe(input_df)

    # ----------------------------------------------
    # Prediction
    # ----------------------------------------------

    try:

        prediction = model.predict(input_df)[0]

        # ------------------------------------------
        # Probability
        # ------------------------------------------

        if hasattr(model, "predict_proba"):

            probability = model.predict_proba(input_df)[0][1]

        else:

            probability = None

        # ------------------------------------------
        # Result
        # ------------------------------------------

        st.subheader("📊 Prediction Result")

        if prediction == 1:

            st.error(
                "⚠️ Employee is likely to LEAVE the company."
            )

        else:

            st.success(
                "✅ Employee is likely to STAY in the company."
            )

        # ------------------------------------------
        # Probability
        # ------------------------------------------

        if probability is not None:

            st.write(
                f"**Attrition Probability: "
                f"{probability * 100:.2f}%**"
            )

            st.progress(float(probability))

    except Exception as e:

        st.error("❌ Prediction failed.")

        st.write("Error:", e)