# import libraries
import io
import pickle
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

# Regression models
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC

# Regression metrics
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

# Classification metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from sklearn.datasets import load_iris


warnings.filterwarnings("ignore")


# PAGE CONFIGURATION

st.set_page_config(
    page_title="Machine Learning App",
    layout="wide",
)


# SESSION STATE

def initialize_session_state():
    """Create variables that need to survive Streamlit reruns."""

    default_values = {
        "data": None,
        "models": {},
        "results": None,
        "best_model_name": None,
        "best_pipeline": None,
        "problem_type": None,
        "feature_columns": [],
        "target_column": None,
        "X_test": None,
        "y_test": None,
        "predictions": None,
    }

    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_session_state()


# DATA LOADING


@st.cache_data
def load_example_dataset(dataset_name):
    """
    Load an example dataset.

    Cached so Streamlit does not load the same dataset repeatedly.
    """

    if dataset_name == "Titanic":
        return sns.load_dataset("titanic")

    elif dataset_name == "Tips":
        return sns.load_dataset("tips")

    elif dataset_name == "Iris":
        # Load Iris using Scikit-Learn
        iris = load_iris()

        df = pd.DataFrame(
            iris.data,
            columns=iris.feature_names
        )

        # Convert target numbers to readable class names
        df["species"] = [
            iris.target_names[i]
            for i in iris.target
        ]

        return df

    return None

# upload the file
 
def load_uploaded_file(uploaded_file):
    """Load CSV, TSV or Excel files."""

    file_name = uploaded_file.name.lower()

    try:

        if file_name.endswith(".csv"):
            return pd.read_csv(uploaded_file)

        elif file_name.endswith(".tsv"):
            return pd.read_csv(
                uploaded_file,
                sep="\t"
            )

        elif file_name.endswith(".xlsx"):
            return pd.read_excel(uploaded_file)

        elif file_name.endswith(".xls"):
            return pd.read_excel(uploaded_file)

        else:
            st.error(
                "Unsupported file format. "
                "Please upload CSV, TSV, XLSX or XLS."
            )

    except Exception as error:
        st.error(f"Could not read the file: {error}")

    return None


# PROBLEM TYPE DETECTION

def detect_problem_type(target):
    """
    Automatically detect whether the target represents
    regression or classification.

    Numeric target with many unique continuous values
    -> Regression

    Otherwise
    -> Classification
    """

    if pd.api.types.is_numeric_dtype(target):

        unique_values = target.nunique()
        total_values = len(target)

        # A numeric target with relatively few unique values
        # is usually classification.
        if unique_values <= 10 or unique_values / total_values < 0.05:
            return "Classification"

        return "Regression"

    return "Classification"


# DATA VALIDATION

def validate_data(df, features, target):
    """Check whether the user's selections are valid."""

    if df is None:
        return False, "Please select or upload a dataset."

    if df.empty:
        return False, "The dataset is empty."

    if not features:
        return False, "Please select at least one feature."

    if target is None:
        return False, "Please select a target column."

    if target in features:
        return False, "Target column cannot also be a feature."

    if target not in df.columns:
        return False, "Selected target column does not exist."

    # Check whether target contains only missing values
    if df[target].isna().all():
        return False, "Target column contains only missing values."

    return True, ""


 # PREPROCESSING PIPELINE

def build_preprocessor(X):
    """
    Build preprocessing pipelines for numerical and
    categorical features.

    Numerical:
        IterativeImputer -> StandardScaler

    Categorical:
        IterativeImputer cannot directly handle strings,
        therefore categorical missing values are filled
        using the most frequent value through SimpleImputer.

        Then OrdinalEncoder converts categories to numbers.
    """

    from sklearn.impute import SimpleImputer

    numeric_columns = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    transformers = []

    # --------------------------------------------------------
    # Numerical pipeline
    # --------------------------------------------------------

    if numeric_columns:

        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    IterativeImputer(
                        random_state=42
                    )
                ),
                (
                    "scaler",
                    StandardScaler()
                )
            ]
        )

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_columns
            )
        )

    # --------------------------------------------------------
    # Categorical pipeline
    # --------------------------------------------------------

    if categorical_columns:

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),
                (
                    "encoder",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1
                    )
                )
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    return preprocessor


# MODELS
 
def get_regression_models():
    """Return available regression models."""

    return {
        "Linear Regression": LinearRegression(),

        "Decision Tree": DecisionTreeRegressor(
            random_state=42
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),

        "Support Vector Machine": SVR()
    }


def get_classification_models():
    """Return available classification models."""

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000
        ),

        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),

        "Support Vector Machine": SVC(
            probability=True,
            random_state=42
        )
    }


 # TRAIN MODELS
 
def train_models(
    X_train,
    X_test,
    y_train,
    y_test,
    selected_models,
    problem_type
):
    """
    Build a separate preprocessing + model pipeline
    for every selected model.
    """

    trained_models = {}
    results = []

    total_models = len(selected_models)

    progress = st.progress(0)

    status = st.empty()

    for index, model_name in enumerate(selected_models):

        status.info(
            f"Training {model_name}..."
        )

        # Create preprocessing pipeline
        preprocessor = build_preprocessor(X_train)

        # Get correct model
        if problem_type == "Regression":
            model = get_regression_models()[model_name]
        else:
            model = get_classification_models()[model_name]

        # Complete pipeline
        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "model",
                    model
                )
            ]
        )

        try:

            # Train
            pipeline.fit(
                X_train,
                y_train
            )

            # Predict
            predictions = pipeline.predict(
                X_test
            )

            #     
            # REGRESSION
            #     

            if problem_type == "Regression":

                mse = mean_squared_error(
                    y_test,
                    predictions
                )

                rmse = np.sqrt(mse)

                mae = mean_absolute_error(
                    y_test,
                    predictions
                )

                r2 = r2_score(
                    y_test,
                    predictions
                )

                results.append(
                    {
                        "Model": model_name,
                        "MSE": mse,
                        "RMSE": rmse,
                        "MAE": mae,
                        "R²": r2
                    }
                )

            #     
            # CLASSIFICATION
            #     

            else:

                accuracy = accuracy_score(
                    y_test,
                    predictions
                )

                precision = precision_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0
                )

                recall = recall_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0
                )

                f1 = f1_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0
                )

                # ROC-AUC

                auc = np.nan

                try:

                    if hasattr(
                        pipeline,
                        "predict_proba"
                    ):

                        probabilities = (
                            pipeline.predict_proba(
                                X_test
                            )
                        )

                        classes = pipeline.classes_

                        if len(classes) == 2:

                            auc = roc_auc_score(
                                y_test,
                                probabilities[:, 1]
                            )

                        else:

                            auc = roc_auc_score(
                                y_test,
                                probabilities,
                                multi_class="ovr",
                                average="weighted"
                            )

                except Exception:
                    pass

                results.append(
                    {
                        "Model": model_name,
                        "Accuracy": accuracy,
                        "Precision": precision,
                        "Recall": recall,
                        "F1 Score": f1,
                        "ROC-AUC": auc
                    }
                )

            trained_models[model_name] = pipeline

        except Exception as error:

            st.warning(
                f"{model_name} could not be trained: "
                f"{error}"
            )

        progress.progress(
            (index + 1) / total_models
        )

    status.success(
        "Model training completed!"
    )

    return trained_models, pd.DataFrame(results)


 # FIND BEST MODEL
 
def find_best_model(results, problem_type):
    """Find the best model using the primary metric."""

    if results.empty:
        return None

    if problem_type == "Regression":

        # Higher R² is better
        best_row = results.loc[
            results["R²"].idxmax()
        ]

    else:

        # Higher F1 Score is better
        best_row = results.loc[
            results["F1 Score"].idxmax()
        ]

    return best_row["Model"]


 # DATA INFORMATION
 
def show_data_information(df):

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(),
        use_container_width=True
    )

    # Dataset statistics

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Rows",
            df.shape[0]
        )

    with col2:
        st.metric(
            "Columns",
            df.shape[1]
        )

    with col3:
        st.metric(
            "Missing Values",
            int(df.isna().sum().sum())
        )

    # Dataset details

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Description",
            "Data Types",
            "Missing Values",
            "Column Names"
        ]
    )

    with tab1:

        st.dataframe(
            df.describe(
                include="all"
            ).T,
            use_container_width=True
        )

    with tab2:

        dtype_df = pd.DataFrame(
            {
                "Column": df.columns,
                "Data Type": df.dtypes.astype(str)
            }
        )

        st.dataframe(
            dtype_df,
            use_container_width=True
        )

    with tab3:

        missing_df = pd.DataFrame(
            {
                "Column": df.columns,
                "Missing Values": df.isna().sum(),
                "Missing %":
                    (
                        df.isna().mean() * 100
                    ).round(2)
            }
        )

        st.dataframe(
            missing_df,
            use_container_width=True
        )

    with tab4:

        st.write(
            list(df.columns)
        )


 # REGRESSION VISUALIZATION
 
def show_regression_plot(
    y_test,
    predictions
):

    st.subheader(
        "Actual vs Predicted"
    )

    fig, ax = plt.subplots()

    ax.scatter(
        y_test,
        predictions,
        alpha=0.7
    )

    # Perfect prediction line
    minimum = min(
        y_test.min(),
        predictions.min()
    )

    maximum = max(
        y_test.max(),
        predictions.max()
    )

    ax.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--"
    )

    ax.set_xlabel(
        "Actual Values"
    )

    ax.set_ylabel(
        "Predicted Values"
    )

    ax.set_title(
        "Actual vs Predicted Values"
    )

    st.pyplot(fig)

    plt.close(fig)


 # CONFUSION MATRIX
 
def show_confusion_matrix(
    pipeline,
    X_test,
    y_test
):

    st.subheader(
        "Confusion Matrix"
    )

    predictions = pipeline.predict(
        X_test
    )

    fig, ax = plt.subplots()

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        ax=ax
    )

    ax.set_title(
        "Confusion Matrix"
    )

    st.pyplot(fig)

    plt.close(fig)


 # DOWNLOAD MODEL
 
def create_pickle_file(model):

    buffer = io.BytesIO()

    pickle.dump(
        model,
        buffer
    )

    buffer.seek(0)

    return buffer


 # PREDICTION DATA LOADING
 
def load_prediction_file(uploaded_file):

    file_name = uploaded_file.name.lower()

    try:

        if file_name.endswith(".csv"):

            return pd.read_csv(
                uploaded_file
            )

        elif file_name.endswith(".tsv"):

            return pd.read_csv(
                uploaded_file,
                sep="\t"
            )

        elif file_name.endswith(
            (".xlsx", ".xls")
        ):

            return pd.read_excel(
                uploaded_file
            )

        else:

            st.error(
                "Unsupported file format."
            )

    except Exception as error:

        st.error(
            f"Could not read prediction file: {error}"
        )

    return None


 # MANUAL PREDICTION INPUT
 
def create_manual_input(
    df,
    feature_columns
):

    input_data = {}

    st.subheader(
        "Enter Prediction Values"
    )

    columns = st.columns(2)

    for index, column in enumerate(
        feature_columns
    ):

        with columns[index % 2]:

            # Numerical column
            if pd.api.types.is_numeric_dtype(
                df[column]
            ):

                median_value = df[column].median()

                if pd.isna(median_value):
                    median_value = 0.0

                input_data[column] = st.number_input(
                    column,
                    value=float(median_value)
                )

            # Categorical column
            else:

                categories = (
                    df[column]
                    .dropna()
                    .unique()
                    .tolist()
                )

                if categories:

                    input_data[column] = st.selectbox(
                        column,
                        categories
                    )

                else:

                    input_data[column] = st.text_input(
                        column
                    )

    return pd.DataFrame(
        [input_data]
    )


 # SIDEBAR
 
st.sidebar.title(
    "Application Settings"
)

st.sidebar.header(
    "1. Select Data Source"
)

data_source = st.sidebar.radio(
    "Choose an option:",
    [
        "Upload Your Own Data",
        "Use Example Dataset"
    ]
)


 # LOAD DATA
 
df = None

if data_source == "Upload Your Own Data":

    uploaded_file = st.sidebar.file_uploader(
        "Upload your dataset",
        type=[
            "csv",
            "tsv",
            "xlsx",
            "xls"
        ]
    )

    if uploaded_file is not None:

        df = load_uploaded_file(
            uploaded_file
        )

else:

    dataset_name = st.sidebar.selectbox(
        "Select Example Dataset",
        [
            "Titanic",
            "Tips",
            "Iris"
        ]
    )

    df = load_example_dataset(
        dataset_name
    )


 # MAIN APPLICATION
 
st.title(
    "Machine Learning Application"
)

st.write(
    """
    Welcome! This application allows you to upload your own
    dataset or use an example dataset, select features and a
    target, train multiple Scikit-Learn machine learning models,
    compare their performance, and make predictions.
    """
)


 # STOP IF NO DATA
 
if df is None:

    st.info(
        "Please select or upload a dataset from the sidebar."
    )

    st.stop()


# Save current dataset in session state
st.session_state.data = df


 # SHOW DATA
 
show_data_information(df)


 # SIDEBAR MODEL SETTINGS
 
st.sidebar.header(
    "2. Machine Learning Settings"
)

     
# Target selection
     

target_column = st.sidebar.selectbox(
    "Select Target Column",
    df.columns
)

     
# Feature selection
     

available_features = [
    column
    for column in df.columns
    if column != target_column
]

feature_columns = st.sidebar.multiselect(
    "Select Feature Columns",
    available_features,
    default=available_features
)


     
# Problem type
     

problem_selection = st.sidebar.selectbox(
    "Select Problem Type",
    [
        "Auto Detect",
        "Regression",
        "Classification"
    ]
)


 # DETERMINE PROBLEM TYPE
 
if problem_selection == "Auto Detect":

    detected_problem = detect_problem_type(
        df[target_column]
    )

else:

    detected_problem = problem_selection


st.info(
    f"### Problem Type: {detected_problem}"
)


 # TRAIN TEST SPLIT
 
train_size = st.sidebar.slider(
    "Training Data (%)",
    min_value=60,
    max_value=90,
    value=80,
    step=5
)

test_size = 1 - train_size / 100


 # MODEL SELECTION
 
if detected_problem == "Regression":

    available_models = list(
        get_regression_models().keys()
    )

else:

    available_models = list(
        get_classification_models().keys()
    )


selected_models = st.sidebar.multiselect(
    "Select Models",
    available_models,
    default=available_models
)


 # RUN ANALYSIS BUTTON
 
st.sidebar.markdown("---")

run_analysis = st.sidebar.button(
    "Run Analysis",
    type="primary",
    use_container_width=True
)


 # TRAINING
 
if run_analysis:

    # Validate selections

    valid, message = validate_data(
        df,
        feature_columns,
        target_column
    )

    if not valid:

        st.error(message)

        st.stop()

    if not selected_models:

        st.error(
            "Please select at least one model."
        )

        st.stop()

    # Prepare X and y

    X = df[feature_columns].copy()

    y = df[target_column].copy()

    # Remove rows where target is missing

    target_missing = y.isna()

    if target_missing.any():

        st.warning(
            f"{target_missing.sum()} rows with missing "
            "target values were removed."
        )

        X = X.loc[~target_missing]
        y = y.loc[~target_missing]

    
    # Validate problem type

    if detected_problem == "Regression":

        if not pd.api.types.is_numeric_dtype(y):

            st.error(
                "Regression requires a numeric target column."
            )

            st.stop()

        if y.nunique() <= 1:

            st.error(
                "Regression requires more than one target value."
            )

            st.stop()

    else:

        if y.nunique() < 2:

            st.error(
                "Classification requires at least two classes."
            )

            st.stop()

    # Train/test split

    try:

        if detected_problem == "Classification":

            X_train, X_test, y_train, y_test = (
                train_test_split(
                    X,
                    y,
                    test_size=test_size,
                    random_state=42,
                    stratify=y
                )
            )

        else:

            X_train, X_test, y_train, y_test = (
                train_test_split(
                    X,
                    y,
                    test_size=test_size,
                    random_state=42
                )
            )

    except Exception as error:

        st.error(
            f"Could not split the data: {error}"
        )

        st.stop()

    # Train models

    with st.spinner(
        "Training machine learning models..."
    ):

        trained_models, results = train_models(
            X_train,
            X_test,
            y_train,
            y_test,
            selected_models,
            detected_problem
        )

    if results.empty:

        st.error(
            "No model was successfully trained."
        )

        st.stop()

    # Find best model

    best_model_name = find_best_model(
        results,
        detected_problem
    )

    best_pipeline = trained_models[
        best_model_name
    ]

    # Save everything to session state

    st.session_state.models = trained_models

    st.session_state.results = results

    st.session_state.best_model_name = (
        best_model_name
    )

    st.session_state.best_pipeline = (
        best_pipeline
    )

    st.session_state.problem_type = (
        detected_problem
    )

    st.session_state.feature_columns = (
        feature_columns
    )

    st.session_state.target_column = (
        target_column
    )

    st.session_state.X_test = X_test

    st.session_state.y_test = y_test

    st.session_state.predictions = (
        best_pipeline.predict(X_test)
    )


 # DISPLAY RESULTS
 
if (
    st.session_state.results is not None
):

    results = st.session_state.results

    problem_type = (
        st.session_state.problem_type
    )

    best_model_name = (
        st.session_state.best_model_name
    )

    best_pipeline = (
        st.session_state.best_pipeline
    )

    X_test = st.session_state.X_test

    y_test = st.session_state.y_test

    predictions = st.session_state.predictions

    #     =======
    # MODEL COMPARISON
    #     =======

    st.header(
        "Model Comparison"
    )

    st.dataframe(
        results.style.format(
            {
                column: "{:.4f}"
                for column in results.columns
                if column != "Model"
            }
        ),
        use_container_width=True
    )

    #     =======
    # BEST MODEL
    #     =======

    st.header(
        "Best Model"
    )

    st.success(
        f"Best Model: **{best_model_name}**"
    )

    # Best model metrics

    best_result = results[
        results["Model"] == best_model_name
    ].iloc[0]

    if problem_type == "Regression":

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "MSE",
            f"{best_result['MSE']:.4f}"
        )

        col2.metric(
            "RMSE",
            f"{best_result['RMSE']:.4f}"
        )

        col3.metric(
            "MAE",
            f"{best_result['MAE']:.4f}"
        )

        col4.metric(
            "R²",
            f"{best_result['R²']:.4f}"
        )

    else:

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Accuracy",
            f"{best_result['Accuracy']:.4f}"
        )

        col2.metric(
            "Precision",
            f"{best_result['Precision']:.4f}"
        )

        col3.metric(
            "Recall",
            f"{best_result['Recall']:.4f}"
        )

        col4.metric(
            "F1 Score",
            f"{best_result['F1 Score']:.4f}"
        )

        if not pd.isna(
            best_result["ROC-AUC"]
        ):

            st.metric(
                "ROC-AUC",
                f"{best_result['ROC-AUC']:.4f}"
            )

    #     =======
    # VISUALIZATION
    #     =======

    st.header(
        "Model Evaluation"
    )

    if problem_type == "Regression":

        show_regression_plot(
            y_test,
            predictions
        )

    else:

        show_confusion_matrix(
            best_pipeline,
            X_test,
            y_test
        )

    #     =======
    # DOWNLOAD MODEL
    #     =======

    st.header(
        "Download Model"
    )

    download_model = st.checkbox(
        "I want to download the trained model"
    )

    if download_model:

        pickle_file = create_pickle_file(
            best_pipeline
        )

        st.download_button(
            label="⬇Download Model (.pkl)",
            data=pickle_file,
            file_name=(
                f"{best_model_name.replace(' ', '_')}"
                "_model.pkl"
            ),
            mime="application/octet-stream",
            use_container_width=True
        )

    #     =======
    # PREDICTION SECTION
    #     =======

    st.header(
        "Make Predictions"
    )

    make_prediction = st.checkbox(
        "I want to make predictions"
    )

    if make_prediction:

        prediction_method = st.radio(
            "Select prediction method:",
            [
                "Enter values manually",
                "Upload prediction file"
            ]
        )

        #     ===
        # MANUAL INPUT
        #     ===

        if prediction_method == (
            "Enter values manually"
        ):

            input_df = create_manual_input(
                df,
                st.session_state.feature_columns
            )

            if st.button(
                "Make Prediction",
                type="primary"
            ):

                try:

                    prediction = (
                        best_pipeline.predict(
                            input_df
                        )
                    )

                    st.subheader(
                        "Prediction Result"
                    )

                    result_df = pd.DataFrame(
                        {
                            "Prediction":
                                prediction
                        }
                    )

                    st.dataframe(
                        result_df,
                        use_container_width=True
                    )

                    # Classification probabilities
                    if (
                        problem_type
                        == "Classification"
                        and hasattr(
                            best_pipeline,
                            "predict_proba"
                        )
                    ):

                        probabilities = (
                            best_pipeline.predict_proba(
                                input_df
                            )
                        )

                        classes = (
                            best_pipeline.classes_
                        )

                        probability_df = pd.DataFrame(
                            probabilities,
                            columns=[
                                f"Probability ({c})"
                                for c in classes
                            ]
                        )

                        st.subheader(
                            "Prediction Probabilities"
                        )

                        st.dataframe(
                            probability_df,
                            use_container_width=True
                        )

                except Exception as error:

                    st.error(
                        f"Prediction failed: {error}"
                    )

        #     ===
        # FILE INPUT
        #     ===

        else:

            prediction_file = st.file_uploader(
                "Upload prediction data",
                type=[
                    "csv",
                    "tsv",
                    "xlsx",
                    "xls"
                ],
                key="prediction_file"
            )

            if prediction_file is not None:

                prediction_df = (
                    load_prediction_file(
                        prediction_file
                    )
                )

                if prediction_df is not None:

                    st.subheader(
                        "Prediction Data"
                    )

                    st.dataframe(
                        prediction_df.head(),
                        use_container_width=True
                    )

                    required_columns = (
                        st.session_state.feature_columns
                    )

                    missing_columns = [
                        column
                        for column in required_columns
                        if column not in prediction_df.columns
                    ]

                    if missing_columns:

                        st.error(
                            "The following required "
                            f"columns are missing: "
                            f"{missing_columns}"
                        )

                    else:

                        prediction_input = (
                            prediction_df[
                                required_columns
                            ]
                        )

                        if st.button(
                            "Make Prediction",
                            type="primary",
                            key="file_prediction"
                        ):

                            try:

                                predictions = (
                                    best_pipeline.predict(
                                        prediction_input
                                    )
                                )

                                result_df = (
                                    prediction_df.copy()
                                )

                                result_df[
                                    "Prediction"
                                ] = predictions

                                st.success(
                                    "Predictions generated successfully!"
                                )

                                st.dataframe(
                                    result_df,
                                    use_container_width=True
                                )

                                # Download predictions
                                csv_data = (
                                    result_df.to_csv(
                                        index=False
                                    )
                                )

                                st.download_button(
                                    "Download Predictions",
                                    csv_data,
                                    "predictions.csv",
                                    "text/csv"
                                )

                            except Exception as error:

                                st.error(
                                    f"Prediction failed: {error}"
                                )


 # FOOTER
 
st.markdown("---")

st.caption(
    "Built with Python, Streamlit, Pandas and Scikit-Learn"
)