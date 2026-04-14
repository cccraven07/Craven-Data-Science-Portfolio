import streamlit as st #To build website
import pandas as pd #To read data

#Importing sklearn ML utilities from sklean library that will be used in building and evaluating ML models later
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

#Importing types of ML models to be used later: Logistic Regression, Decision Tree Classifier, and K-Nearest Neighbors
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

#For plotting
import matplotlib.pyplot as plt
import seaborn as sns


# -----------------------------
# APP TITLE
# -----------------------------
st.title("Machine Learning App")

st.write(
    "This app allows users to train and evaluate machine learning models, "
    "adjust hyperparameters, and interpret model performance using real datasets."
)


# -----------------------------
# Step 1: Choosing the Data Source
# -----------------------------

#In this model, I have included 3 basic data sources that users can pick from
#These dataframes were fround on kaggle and should be useful for running the basic supervised ML models that this app uses
#I have also included an option for users to upload their own csv file if they choose

st.sidebar.header("Select Data Source") #Titling section

#Selecting a data source
data_option = st.sidebar.selectbox(
    "Choose a dataset",
    ["Student Performance", "Titanic", "Telco Churn", "Upload Your Own Data"]
)

df = None

#Student exam performance dataset, which includes a number of metrics that influence student success
if data_option == "Student Performance":
    df = pd.read_csv("MLStreamlitApp/student_exam_performance_dataset.csv")

#The classic Titanic dataset, including data about passengers on the Titanic, and whether or not they survived
elif data_option == "Titanic":
    df = pd.read_csv("MLStreamlitApp/Titanic-Dataset.csv")

#The Telco Custormer Churn dataset, which includes a number of customer demographics to predict whether or not a customer is likley to leave a telecom company 
elif data_option == "Telco Churn":
    df = pd.read_csv("MLStreamlitApp/WA_Fn-UseC_-Telco-Customer-Churn.csv")

#If users want to, they can upload their own data
elif data_option == "Upload Your Own Data":
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

# Suggested default target variable based on dataset
default_target = None

if data_option == "Student Performance":
    default_target = "pass_fail"

elif data_option == "Titanic":
    default_target = "Survived"

elif data_option == "Telco Churn":
    default_target = "Churn"

#Showing the data that a user chooses
if df is not None:
    st.subheader("Dataset Preview")
   
    #Descriptions for Sample Datasets
    if data_option == "Student Performance":
        st.info("""
        **Student Performance Dataset:**  
        Contains academic and demographic information about students.  
        The goal is usually to predict whether a student will **pass or fail** based on study habits, attendance, and related factors.
        """)

    elif data_option == "Titanic":
        st.info("""
        **Titanic Dataset:**  
        Contains information about passengers on the Titanic (age, sex, class, etc.).  
        The main prediction task is whether a passenger **survived or not**.
        """)

    elif data_option == "Telco Churn":
        st.info("""
        **Telco Customer Churn Dataset:**  
        Contains customer account and service data from a telecom company.  
        The goal is to predict whether a customer will **churn (leave the company)**.
        """)
    st.write(df.head())

    st.subheader("Shape")
    rows, cols = df.shape
    st.write(f"Rows: {rows}, Columns: {cols}")


# -----------------------------
# Step 2: Model Selection
# -----------------------------
if df is not None:

    st.header("Model Training") #Labeling Training and setup section
    st.subheader("Model Setup")

    st.write("""
    ##### What you're doing here
    You are selecting:
    - a **target variable** (what you want to predict)
    - **features** (the inputs used for prediction)
    - a **machine learning model**

    The app will then train the model and evaluate how well it predicts the target.
    """)

    #Creating a default target for the sample dataset I added to the app
    st.write(f"Suggested target for this dataset: **{default_target}**")

    #Selecting a target variable for the model to predict
    target = st.selectbox("Select target variable", 
                          df.columns, 
                          index=df.columns.get_loc(default_target) if default_target in df.columns else 0,
                          help= "This is the variable your model will try to predict")

    #Selecting feature variables that the model will use to predict the target
    #To keep the models simple, users can only select up to 5 feature variables
    features = st.multiselect(
        "Select feature variables (max 5)",
        df.columns.drop(target),
        max_selections=5,
        help="These are the input variables used to predict the target"
    )
    
    #Choosing the type of ML model to use
    model_type = st.selectbox(
        "Choose model",
        ["Logistic Regression", "KNN", "Decision Tree"],
        help="Choose a machine learning algorithm to train your data"
    )

    #Descriptions of the types of models to give users a basic idea of what each model does
    if model_type == "Logistic Regression":
        st.info("""
        **Logistic Regression:** A statistical model used for classification tasks.  
        Best for: binary classification and interpretable relationships between features and outcomes.
        """)

    elif model_type == "KNN":
        st.info("""
        **K-Nearest Neighbors (KNN):** A non-parametric, distance-based model.  
        It classifies a data point based on the majority class of its 'k' closest neighbors in feature space.  
        Best for: classification tasks (binary or multi‑class) where outcomes are categorical.
        """)

    elif model_type == "Decision Tree":
        st.info("""
        **Decision Tree:** Model that makes decisions by asking a series of yes/no questions.  
        Best for: interpretable models and capturing non-linear relationships.
        """)

    # -----------------------------
    # Step 3: Adjusting Hyperparameters
    # -----------------------------
    st.write("### Hyperparameter Tuning")
    st.write("Adjust these settings to control how the model learns from data.")

    #Describing what the hyperparameters are for users to get a basic idea of how they work
    #For each of the ML model types, users can adjust one hyperparameter on a slider
    if model_type == "Logistic Regression":

        st.info("""
        **C (Regularization Strength):** Controls how strongly the model is penalized for large coefficients.  
        - Low C → stronger regularization (simpler model, less overfitting)  
        - High C → weaker regularization (more flexible model, may overfit)
        """)

        st.write("Note that this slider is log scaled. For example, a value of 0 on the slider will give a C value of 1.")

        # Log-scale slider for C
        log_C = st.slider(
            "Log10(C) (Regularization Strength)",
            -2.0, 2.0, 0.0
        )

        C_value = 10 ** log_C

        st.write(f"Actual C value: {C_value:.4f}")

    elif model_type == "KNN":

        st.info("""
        **K (Number of Neighbors):** The number of nearby data points used to classify a new point.  
        - Small K → more sensitive, may overfit  
        - Large K → smoother decision boundary, more stable but less flexible
        """)

        k = st.slider("Number of Neighbors (k)", 1, 15, 5)

    elif model_type == "Decision Tree":

        st.info("""
        **Max Depth:** The maximum number of splits in the tree.  
        - Low depth → simpler tree, less overfitting  
        - High depth → more complex tree, may overfit the data
        """)

        max_depth = st.slider("Max Depth", 1, 20, 5)

    #After variables are set, users can train the model
    train = st.button("Train Model")


    # -----------------------------
    # Step 4: Train Model
    # -----------------------------
    if train and len(features) > 0:

        #Split data into feature and testing variables
        X = df[features]
        y = df[target]

        X = pd.get_dummies(X, drop_first=True) #to convert categorical variables to numerics
        st.session_state["feature_names"] = X.columns #saving feature names so they can be displayed and interpreted
        X = X.fillna(X.mean(numeric_only=True)) #handling missing values simply by replacing with means

        #Encode target lables if they are categorical
        if y.dtype == "object":
            y = y.astype("category").cat.codes

        #Splitting data into training and testing sets, using the standard 80/20 model
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        #Initializes scalare for KNN
        scaler = None

        #Scales the data for KNN model to ensure variables on the same scale
        if model_type == "KNN":
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        #Executing each of the models based off of user input
        if model_type == "Logistic Regression":
            model = LogisticRegression(C=C_value, max_iter=1000)

        elif model_type == "KNN":
            model = KNeighborsClassifier(n_neighbors=k)

        else:
            model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)

        model.fit(X_train, y_train) #training the model based on the training data
        preds = model.predict(X_test) #generating the predictions based on the test data

        #Saving these variables so they can be later used in visualizations
        st.session_state["model"] = model
        st.session_state["X_test"] = X_test
        st.session_state["y_test"] = y_test
        st.session_state["preds"] = preds

        #Displaying training success message and accuracy score
        st.success(f"{model_type} trained!")
        acc = accuracy_score(y_test, preds)
        st.write("Accuracy:", accuracy_score(y_test, preds))
        st.info(f"This means the model correctly predicted about {acc:.0%} of cases.") #intuitively interpreting accuracy


# -----------------------------
# Step 5: Visualizing model success on various metrics
# -----------------------------

st.header("Model Evaluation") #labeling section

#This section retrieves stored predictions and compares them to true labels
if "model" in st.session_state:

    #Recalling the variables from above
    model = st.session_state["model"]
    X_test = st.session_state["X_test"]
    y_test = st.session_state["y_test"]

    # ---------------------------------
    # A. Generating a Confusion Matrix
    # ---------------------------------
    st.subheader("Confusion Matrix") #section title

    st.info(
            "The confusion matrix shows how well the model performed by comparing "
            "predicted labels to actual labels. "
            "Correct predictions appear on the diagonal (top left and bottom right), while mistakes appear off-diagonal. "
            "It is especially useful for understanding *types* of errors the model makes."
    )

    # Default predictions fallback (in case no probability model)
    preds_final = None

    if hasattr(model, "predict_proba"):

        probs = model.predict_proba(X_test)

        if probs.shape[1] == 2:

            #Adding a decision threshold slider for users to adjust
            threshold = st.slider(
                "Decision Threshold",
                0.1, 0.9, 0.5
            )

            st.info(
                "The decision threshold determines how the model converts probabilities into class predictions.\n\n"
                "- If predicted probability ≥ threshold → classify as positive (1)\n"
                "- If predicted probability < threshold → classify as negative (0)\n\n"
                "Default is 0.5, but changing it allows you to control the trade-off between false positives and false negatives.\n"
                "Lower thresholds catch more positives (fewer false negatives). Higher thresholds are more strict (fewer false positives)."
            )

            # Convert probabilities → predictions using threshold
            y_scores = probs[:, 1]
            preds_final = (y_scores >= threshold).astype(int)

    # If model does NOT support probabilities, fallback to normal predictions
    if preds_final is None:
        preds_final = model.predict(X_test)

    #Generating a confusion matrix based on the data to visualize model success
    cm = confusion_matrix(y_test, preds_final)

    #Color coding the confusion matrix
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    #Adding another description to help users interpret the confusion matrix and readjust parameters
    st.write(
        "In some real-world applications, the cost of different errors is not equal. "
        "For example, in medical screening (like cancer detection), it is often better to have "
        "more false positives (flagging healthy patients as positive) than false negatives "
        "(missing actual cases), because missing a positive case can be dangerous. "
        "The confusion matrix helps you evaluate this tradeoff."
    )

    # ---------------------
    # B. ROC and AUC Curves
    # ---------------------

    # ROC (Receiver Operating Characteristic) curve evaluates how well the model
    # separates the two classes across ALL possible decision thresholds.

    if hasattr(model, "predict_proba"):

        probs = model.predict_proba(X_test)

        if probs.shape[1] == 2:

            st.subheader("ROC Curve")

            st.info(
            "The ROC curve shows how well the model separates the two classes "
            "across all possible decision thresholds (0 to 1). "
            "A threshold (commonly 0.5) determines how probabilities are converted into predictions:\n\n"
            "- Above 0.5 → predict positive class (1)\n"
            "- Below 0.5 → predict negative class (0)\n\n"
            "Each point on the ROC curve represents a different threshold."
            )

            st.write(
            "The diagonal orange line represents a completely random model (no predictive power). "
            "A good model curves upward toward the top-left corner, meaning it achieves high true positives "
            "while keeping false positives low."
            )

            #Probability of positive class
            y_scores = probs[:, 1]

            #Computing ROC
            fpr, tpr, _ = roc_curve(y_test, y_scores)
            roc_auc = auc(fpr, tpr)

            #Plotting the Graph
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
            ax.plot([0, 1], [0, 1], linestyle="--")

            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend()

            st.pyplot(fig)

            st.write("AUC Score:", roc_auc)

            #Interpreting the meaning of the graph
            st.write(
            f"The AUC score is {roc_auc:.3f}. "
            "This measures how well the model can tell the two classes apart.\n\n"
            "A higher score means the model is better at correctly ranking positive cases above negative ones.\n"
            "A score of 0.5 means the model is no better than random guessing, while 1.0 means perfect separation."
            )

        # ---------------------------
        # C. Prediction Probabilities
        # ---------------------------
        #This section shows the model’s confidence for each test example.
        #Each row corresponds to one observation from the test dataset.

        st.subheader("Prediction Probabilities") #title

        #Adding a description for app users about what the table means
        st.info(
            "This table shows how confident the model is in its predictions for each test example.\n\n"
            "- Each row represents one person/observation from the dataset\n"
            "- Each column shows the probability of belonging to a class\n"
            "- The model predicts the class with the higher probability\n\n"
            "Example: If Class 1 = 0.80, the model is 80% confident the outcome is Class 1."
        )

        #Generating a dataframe
        prob_df = pd.DataFrame(
            probs,
            columns=[f"Class {i}" for i in range(probs.shape[1])]
        )

        st.write(prob_df.head())


    # ---------------------
    # D. Feature Importance
    # ---------------------
    st.subheader("Model Interpretability") #section title

    if isinstance(model, LogisticRegression):

        # Logistic Regression learns a linear relationship between features and the target.
        # Each feature is assigned a coefficient (weight) that represents its impact on the prediction.

        #Retrieves variable names
        feature_names = st.session_state.get("feature_names", None)

        #Generates feature and coefficient categories for the table 
        if feature_names is not None:

            coef_df = pd.DataFrame({
                "Feature": feature_names,
                "Coefficient": model.coef_[0]
            })

            #Sorts features by absolute value of coefficient (to emphasize coefficient strength)
            coef_df = coef_df.sort_values(by="Coefficient", key=abs, ascending=False)

            #Table description
            st.info(
            "This table shows how each input feature affects the model’s prediction.\n\n"
            "- Features = the input variables used by the model (e.g., age, income, score)\n"
            "- Coefficient = how strongly and in what direction each feature influences the prediction\n\n"
            "Positive values push the prediction toward Class 1, negative values push toward Class 0.\n"
            "Larger values (in magnitude) mean the feature has a stronger effect."
            )

            #Plotting
            st.dataframe(coef_df.head(20))

            ### Generating a Bar Chart ###

            #In case of many feature variables, select top ones for bar chart visualization
            top_features = coef_df.head(10)

            st.write("### Visualizing the Most Important Features")

            st.info(
                "The bar chart below shows the top features ranked by importance. "
                "Longer bars mean the feature has a stronger influence on the model’s prediction."
            )

            #Plotting
            fig, ax = plt.subplots()
            bars = ax.barh(top_features["Feature"], top_features["Coefficient"])

            ax.invert_yaxis()

            #Add coefficient labels next to bars
            for i, v in enumerate(top_features["Coefficient"]):
                ax.text(v, i, f"{v:.2f}", va='center')

            st.pyplot(fig)

    #Feature importance for decision trees
    elif isinstance(model, DecisionTreeClassifier):

        #Decision Trees determine which features are most useful by measuring how much
        #each feature helps split the data and reduce prediction error.

        #Retrieve feature names like before
        feature_names = st.session_state.get("feature_names", None)

        if feature_names is not None:

            #Create Dataframe of feature importances
            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": model.feature_importances_
            })

            #Sort features by importance so highest is first 
            importance_df = importance_df.sort_values(by="Importance", ascending=False)

            #Describing what the the values of the table mean
            st.info(
            "This table shows how important each feature is in making predictions.\n\n"
            "- Features = input variables used by the model\n"
            "- Importance = how much that feature helped improve the model’s decisions\n\n"
            "Features with higher importance values were used more often and had a bigger impact.\n"
            "A value close to 0 means the feature was not very useful."
            )

            #Plotting
            st.dataframe(importance_df.head(20))

            ### Generating a Bar Chart ###

            #Selecting top features, as before 
            top_features = importance_df.head(10)

            st.write("### Visualizing the Most Important Features")

            #Bar Chart Descriptor
            st.info(
                "The bar chart below shows the top features ranked by importance. "
                "Longer bars indicate features that had a greater impact on the model’s decisions."
            )

            # Create horizontal bar chart
            fig, ax = plt.subplots()
            bars = ax.barh(top_features["Feature"], top_features["Importance"])

            ax.invert_yaxis()

            # Add importance values next to bars
            for i, v in enumerate(top_features["Importance"]):
                ax.text(v, i, f"{v:.2f}", va='center')

            st.pyplot(fig)
    else:

        #KNN does not have any sort of feature importance because it only makes predictions by finding nearest data point
        
        #Description of why KNN does not have feature importance in the graph
        st.write("K-Nearest Neighbors (KNN) does not provide feature importance because it does not learn "
        "a mathematical model. Instead, it makes predictions by comparing new data points to the most "
        "similar examples in the dataset. Since it does not assign weights to features, there is no direct "
        "measure of feature importance.")