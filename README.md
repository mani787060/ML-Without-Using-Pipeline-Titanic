# Machine Learning Without Using Pipelines: Manual Workflows on Titanic
[![Machine Learning](https://img.shields.io/badge/Domain-Feature%20Engineering-blue)](https://scikit-learn.org/)
[![Architecture-Manual](https://img.shields.io/badge/Workflow-Manual%20Slicing-red)](#)
[![Dataset](https://img.shields.io/badge/Dataset-Titanic%20Survival-green)](#-dataset-profile)

This repository provides a deep-dive, architectural look at manual data preprocessing, vectorization, and modeling without using Scikit-Learn pipelines. Using the historical `Titanic-Dataset.csv`, this project decomposes every stage of a machine learning workflow down to its fundamental operations. By explicitly tracking states, managing missing indices, and manually transforming features, this repository serves as a prerequisite learning layer to understand the hidden mechanics that pipelines abstract away.

---

## 🏗️ The Manual Engineering Framework

When training an estimator without pipeline orchestration, the engineer must act as the manual centralized scheduler. This project tracks the exact algorithmic order of operations required to transition raw, multi-type data into clean, mathematically balanced NumPy matrices while rigorously defending against **Data Leakage**.


                           ┌──> Age, Fare ──────> Mean/Median Imputation ──> StandardScaler  ──┐
                           │                                                                   │
Raw Titanic Data Arrays ───┼──> Sex, Embarked ──> Most Frequent Imputation ──> OneHotEncoder ──┼──> np.hstack() ──> DecisionTree/RandomForest
                           │                                                                   │
                           └──> PassengerId ────> Feature Drop (Irrelevant Noise) ─────────────┘


### 1. Hardcoded Dependency & Sequence Mechanics
*   **The Data Leakage Vulnerability:** Performing transformations like computing the global mean of the `Age` column *before* executing the train-test split is a massive anti-pattern. Doing so leaks structural information from the validation distribution into the training scope. This notebook handles this by splitting raw slices first, computing statistical vectors (`mean`, `median`, `mode`) *solely* from the training subset, and manually mapping them across the test space.
*   **Dimensional Re-Stitching:** Because manual preprocessing isolates and drops specific columns or arrays temporarily, features must be transformed independently. Continuous inputs are passed through a scaling engine, and nominal variables are routed to a categorical encoder. The resulting separate array segments must then be combined using multi-dimensional horizontal stacking (`np.hstack`), meaning you must carefully track structural row dimensions across every step.

---

## 🛠️ Implementation & Technical Workflow

The `ML-Without-Using-Pipeline-Titanic.ipynb` notebook maps out the following operational phases:

1.  **Data Cleaning & Dimensional Selection:** 
    *   Sifting out high-cardinality noise identifiers that contain no generalized predictive capacity (e.g., `PassengerId`, `Name`, `Ticket`).
2.  **Explicit Structural Imputation:** 
    *   Isolating the training slice to calculate imputation baselines.
    *   Replacing missing values in `Age` using median distributions and `Embarked` using mode parameters.
3.  **Nominal Categorical Encoding:** 
    *   Transforming strings like `Sex` and `Embarked` into numerical dimensions via `OneHotEncoder(drop='first', sparse_output=False)`.
    *   Explicitly managing the creation of $k-1$ features to ensure linear model compatibility.
4.  **Feature Scaling & Normalization:**
    *   Applying `StandardScaler` on high-variance numerical elements (`Age`, `Fare`) to center variances around a zero-mean state ($1\sigma$).
5.  **Matrix Concentration:**
    *   Merging the standalone modified arrays back into a uniform matrix array using `np.hstack` or `pd.concat` to prepare it for estimator training.
6.  **Model Fitting & Evaluation:**
    *   Passing the final engineered matrix directly into a downstream classification algorithm to compute baseline classification metrics.

---

## 📊 Core Concepts & Preprocessing Trade-offs

| Manual Preprocessing Step | Architectural Objective | Downstream Impact |
| :--- | :--- | :--- |
| **Train-Test Split Priority** | Prevent structural leakage | Guarantees authentic validation scoring |
| **Isolated Feature Imputation** | Resolve missing matrix values | Maintains shape without injecting future data |
| **$(k-1)$ Nominal OHE** | Evade the Dummy Variable Trap | Prevents multi-collinearity failures in linear models |
| **Explicit Array Stacking** | Matrix reconstruction | Unifies disparate continuous and discrete vectors |

> ⚠️ **The Maintenance Trade-Off:** Operating completely without pipelines highlights how easily a workflow can break. If any feature order is flipped during testing, or if an extra column is added to the raw source data, a manual row-stitching architecture will fail silently or throw index alignment exceptions. This highlights exactly why production applications rely on `Pipeline` and `ColumnTransformer` modules.

---

## 💻 Tech Stack & Requirements
*   **Language Ecosystem:** Python 3.9+
*   **Data Layout Engines:** Pandas, NumPy
*   **Machine Learning Architecture:** Scikit-Learn (`preprocessing`, `model_selection`)
*   **Development Interface:** Jupyter Notebook Execution Engine

---

## 🚀 Getting Started & Execution

1.  **Clone the Architecture:**
    ```bash
    git clone [https://github.com/your-username/ML-Without-Using-Pipeline-Titanic.git](https://github.com/your-username/ML-Without-Using-Pipeline-Titanic.git)
    cd ML-Without-Using-Pipeline-Titanic
    ```

2.  **Install Essential Dependencies:**
    ```bash
    pip install pandas numpy scikit-learn jupyter
    ```

3.  **Execute the Interactive Notebook:**
    ```bash
    jupyter notebook
    ```
    Open and run `ML-Without-Using-Pipeline-Titanic.ipynb` sequentially to analyze the manual array transformations, handle empty values, and observe how data flows step-by-step before implementing model optimization layers.

