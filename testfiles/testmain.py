import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, r2_score, mean_absolute_error

st.title("OptiMLFlow")

# Sidebar for data upload
with st.sidebar:
    st.header("Upload Data")
    uploaded_file = st.file_uploader("Upload dataset", type=["csv", "data", "txt"])
    
    if uploaded_file:
        # Add file type detection and handling
        file_ext = uploaded_file.name.split('.')[-1].lower()
        
        if file_ext == 'data':
            # Try to detect separator for .data files
            sample = uploaded_file.read(2048).decode('utf-8')
            uploaded_file.seek(0)
            
            # Detect most common separator in first few lines
            first_lines = sample.split('\n')[:5]
            possible_seps = [',', ';', '\t', ' ']
            sep_counts = {sep: sum(line.count(sep) for line in first_lines) for sep in possible_seps}
            detected_sep = max(sep_counts, key=sep_counts.get)
            
            sep = st.text_input("Detected separator. Change if needed:", detected_sep)
        else:
            sep = st.text_input("CSV separator (default ',')", ",")
    
    test_size = st.slider("Test set size (fraction)", 0.1, 0.5, 0.2, step=0.05)
    task_type = st.selectbox("Task type", ["Auto", "Classification", "Regression"])
    selected_models_cls = st.multiselect(
        "Classification Models",
        ["RandomForestClassifier", "GradientBoostingClassifier", "LogisticRegression"],
        ["RandomForestClassifier", "LogisticRegression"]
    )
    selected_models_reg = st.multiselect(
        "Regression Models",
        ["RandomForestRegressor", "GradientBoostingRegressor", "LinearRegression"],
        ["RandomForestRegressor", "LinearRegression"]
    )
    selected_metric_cls = st.selectbox("Classification Metric", ["accuracy", "f1", "roc_auc"])
    selected_metric_reg = st.selectbox("Regression Metric", ["r2", "mae"])
    hyperparameter_tuning = st.checkbox("Enable Hyperparameter Tuning")
    cv_fold_option = st.selectbox("CV folds", ["Auto"] + [str(i) for i in range(2, 11)], index=0)
    train_button = st.button("Train Models and Recommend Best")

# Main content area
if uploaded_file is not None:
    try:
        # Add error handling for file reading
        df = pd.read_csv(uploaded_file, sep=sep, on_bad_lines='warn')
        
        # Check if data was loaded properly
        if df.empty:
            st.error("The uploaded file is empty or couldn't be read properly.")
            st.stop()
            
        # Display file info
        st.write(f"File type: .{uploaded_file.name.split('.')[-1]}")
        st.write(f"Number of rows: {df.shape[0]}")
        st.write(f"Number of columns: {df.shape[1]}")
        
        # Display any parsing warnings
        if df.shape[1] == 1:
            st.warning("Only one column detected. Please check if the separator is correct.")
        
        st.write("Dataset Preview:")
        st.dataframe(df.head())
        
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        st.error("Please check if the file format and separator are correct.")
        st.stop()

    target_col = st.selectbox("Select target column (y)", df.columns)
    X = df.drop(columns=[target_col])
    y = df[target_col].copy()

    id_cols = [col for col in X.columns if X[col].nunique() == len(X)]
    if id_cols:
        X = X.drop(columns=id_cols)
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=[object, 'category', 'bool']).columns.tolist()
    high_card_cols = [col for col in cat_cols if X[col].nunique() > 50]
    low_card_cols = [col for col in cat_cols if col not in high_card_cols]
    scaler_choice = 'StandardScaler' if (num_cols and X[num_cols].var().max() > 1) else 'MinMaxScaler'
    transformers = []

    if num_cols:
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler() if scaler_choice == 'StandardScaler' else MinMaxScaler())
        ])
        transformers.append(('num', num_pipeline, num_cols))
    if low_card_cols:
        cat_low_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        transformers.append(('cat_low', cat_low_pipeline, low_card_cols))
    if high_card_cols:
        cat_high_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder())
        ])
        transformers.append(('cat_high', cat_high_pipeline, high_card_cols))

    preprocessor = ColumnTransformer(transformers, remainder='drop')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    X_train_proc = pd.DataFrame(preprocessor.fit_transform(X_train))
    X_test_proc = pd.DataFrame(preprocessor.transform(X_test))

    st.download_button(
        "Download Processed Train Data",
        pd.concat([X_train_proc, y_train.reset_index(drop=True)], axis=1).to_csv(index=False).encode('utf-8'),
        "train_processed.csv", "text/csv"
    )
    st.download_button(
        "Download Processed Test Data",
        pd.concat([X_test_proc, y_test.reset_index(drop=True)], axis=1).to_csv(index=False).encode('utf-8'),
        "test_processed.csv", "text/csv"
    )

    if train_button:
        models_classification = {
            "RandomForestClassifier": RandomForestClassifier(random_state=42),
            "GradientBoostingClassifier": GradientBoostingClassifier(random_state=42),
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42)
        }
        models_regression = {
            "RandomForestRegressor": RandomForestRegressor(random_state=42),
            "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42),
            "LinearRegression": LinearRegression()
        }
        param_grids_cls = {
            "RandomForestClassifier": {'n_estimators': [100, 200], 'max_depth': [None, 5, 10]},
            "GradientBoostingClassifier": {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1]},
            "LogisticRegression": {'C': [0.1, 1, 10]}
        }
        param_grids_reg = {
            "RandomForestRegressor": {'n_estimators': [100, 200], 'max_depth': [None, 5, 10]},
            "GradientBoostingRegressor": {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1]},
            "LinearRegression": {}
        }

        auto_task = ((y.dtype == object) or (y.nunique() <= 20 and y.dtype != float))
        if task_type == "Auto":
            current_task = "classification" if auto_task else "regression"
        else:
            current_task = task_type.lower()

        # CV folds logic with skipping CV if data too small
        if cv_fold_option == "Auto":
            if current_task == "classification":
                min_class_count = y_train.value_counts().min()
                cv_folds = min(10, min_class_count)
                if cv_folds < 2:
                    st.warning("Not enough samples per class for cross-validation. Skipping CV and using single train/test split.")
                    use_cv = False
                else:
                    use_cv = True
            else:
                cv_folds = min(10, len(y_train))
                if cv_folds < 2:
                    st.warning("Not enough samples for cross-validation. Skipping CV and using single train/test split.")
                    use_cv = False
                else:
                    use_cv = True
        else:
            cv_folds = int(cv_fold_option)
            if cv_folds < 2:
                st.warning("CV folds must be at least 2. Setting to 2.")
                cv_folds = 2
            use_cv = True

        if current_task == "classification":
            models_to_train = {k: models_classification[k] for k in selected_models_cls}
            selected_metric = selected_metric_cls
            metric_func_map = {
                "accuracy": accuracy_score,
                "f1": f1_score,
                "roc_auc": roc_auc_score
            }
        else:
            models_to_train = {k: models_regression[k] for k in selected_models_reg}
            selected_metric = selected_metric_reg
            metric_func_map = {
                "r2": r2_score,
                "mae": mean_absolute_error
            }

        performance = {}
        for name, model in models_to_train.items():
            if hyperparameter_tuning:
                param_grid = param_grids_cls[name] if current_task == "classification" else param_grids_reg[name]
                if use_cv:
                    gs = GridSearchCV(model, param_grid, cv=cv_folds, scoring=selected_metric)
                    gs.fit(X_train_proc, y_train)
                    best_estimator = gs.best_estimator_
                else:
                    best_estimator = model.fit(X_train_proc, y_train)
            else:
                best_estimator = model.fit(X_train_proc, y_train)

            y_pred = best_estimator.predict(X_test_proc)

            if current_task == "classification":
                if selected_metric == "roc_auc":
                    if hasattr(best_estimator, "predict_proba"):
                        y_pred_score = best_estimator.predict_proba(X_test_proc)[:, 1]
                    else:
                        y_pred_score = y_pred
                    score = metric_func_map[selected_metric](y_test, y_pred_score)
                elif selected_metric == "f1":
                    score = metric_func_map[selected_metric](y_test, y_pred, average="weighted")
                else:
                    score = metric_func_map[selected_metric](y_test, y_pred)
            else:
                score = metric_func_map[selected_metric](y_test, y_pred)

            if use_cv:
                cv_score = cross_val_score(best_estimator, X_train_proc, y_train, cv=cv_folds, scoring=selected_metric).mean()
            else:
                cv_score = float('nan')

            performance[name] = (score, cv_score)

        best_model = max(performance, key=lambda k: performance[k][0])

        st.subheader("Model Performance:")
        for model_name, (score, cv_score) in performance.items():
            if use_cv:
                st.write(f"{model_name}: Test {selected_metric} = {score:.4f} | CV {selected_metric} = {cv_score:.4f}")
            else:
                st.write(f"{model_name}: Test {selected_metric} = {score:.4f} | CV skipped due to small data")
        st.write(f"Recommended Model: {best_model}")

        # After st.subheader("Model Performance:")
        perf_df = pd.DataFrame([
            {"Model": k, "Test Score": v[0], "CV Score": v[1]} for k, v in performance.items()
        ])

        fig = px.bar(
            perf_df, x="Model", y="Test Score", 
            color="Model", 
            title="Test Performance by Model",
            text="Test Score"
        )
        st.plotly_chart(fig, use_container_width=True)

        if use_cv:
            fig_cv = px.bar(
                perf_df, x="Model", y="CV Score", 
                color="Model", 
                title="Cross-Validation Performance by Model",
                text="CV Score"
            )
            st.plotly_chart(fig_cv, use_container_width=True)

    st.subheader("Feature Distribution")
    selected_feature = st.selectbox("Select feature to plot", X.columns)
    fig_feat = px.histogram(df, x=selected_feature, nbins=30, title=f"Distribution of {selected_feature}")
    st.plotly_chart(fig_feat, use_container_width=True)
