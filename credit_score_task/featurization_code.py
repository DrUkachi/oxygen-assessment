## Libraries for First Part of Pipeline
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted # To check if fitted
from collections import defaultdict
from scipy.stats import chi2_contingency
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted # To check if fitted
from collections import defaultdict, OrderedDict
from scipy.stats import chi2_contingency

# Libraries for the second part of the Pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
import xgboost as xgb
import joblib # For saving/loading the pipeline


def prepare_imputation_strategy(train_df, target_column='default'):
    """
    Analyzes training data to determine imputation strategy.
    Calculates grouped (by target) and global (overall) imputation values
    for columns where missingness is significantly associated with the target.
    Args:
        train_df (pd.DataFrame): The training dataframe.
        target_column (str): The name of the target variable column.
    Returns:
        tuple: Contains:
            - imputation_instructions (dict): {col: 'must-impute'/'not-necessary'}
            - grouped_imputation_values (dict): {col: {target_val: median/mode}}
            - global_imputation_values (dict): {col: global_median/mode}
            - columns_to_impute (list): List of column names needing imputation.
            - missing_indicator_cols (list): List of names for the missing indicator columns to be created.
    """
    df = train_df.copy() # Work on a copy
    null_info = df.isnull().sum().reset_index()
    imputation_instructions = {} # {col: 'must-impute'/'not-necessary'}
    grouped_imputation_values = defaultdict(dict) # {feature: {class_value: median/mode}}
    global_imputation_values = {} # {feature: global_median/mode}
    columns_to_impute = []
    missing_indicator_cols = []

    print("--- Preparing Imputation Strategy (within fit) ---")

    for i in range(len(null_info)):
        column_name = null_info.iloc[i, 0]
        missing_count = null_info.iloc[i, 1]

        if missing_count > 0 and column_name != target_column:
            # print(f"Analyzing column: {column_name} (Missing: {missing_count})") # Reduce verbosity in pipeline
            missing_col_name = f"{column_name}_missing"
            missing_indicator = df[column_name].isna()
            p_value = 1.0
            try:
                contingency_table = pd.crosstab(missing_indicator, df[target_column])
                if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2 or contingency_table.min().min() == 0:
                   # print(f"  Skipping Chi-squared test for {column_name} due to insufficient data variation.")
                   if missing_count / len(df) > 0.05: # Impute if > 5% missing anyway
                       # print(f"  Marking {column_name} for imputation due to missing count > 5%.")
                       p_value = 0.0
                   else:
                       p_value = 1.0
                else:
                    chi2, p_value, _, expected_freq = chi2_contingency(contingency_table)
                    # if (expected_freq < 5).any().any():
                    #      print(f"  Warning: Chi-squared test for {column_name} has expected frequencies < 5.")
            except ValueError as e:
                 # print(f"  Error during Chi-squared test for {column_name}: {e}. Treating as 'must-impute'.")
                 p_value = 0.0

            if p_value < 0.05:
                # print(f"  '{column_name}' missingness IS significantly associated. Marking for imputation.")
                imputation_instructions[column_name] = 'must-impute'
                columns_to_impute.append(column_name)
                missing_indicator_cols.append(missing_col_name)
                is_numeric = pd.api.types.is_numeric_dtype(df[column_name])

                if is_numeric:
                    global_value = df[column_name].median()
                    group_values = df.groupby(target_column)[column_name].median().to_dict()
                else:
                    mode_series = df[column_name].mode()
                    global_value = mode_series.iloc[0] if not mode_series.empty else None
                    group_modes = {}
                    for name, group_df in df.groupby(target_column):
                         mode_series_group = group_df[column_name].mode()
                         group_modes[name] = mode_series_group.iloc[0] if not mode_series_group.empty else global_value
                    group_values = group_modes

                global_imputation_values[column_name] = global_value
                grouped_imputation_values[column_name] = group_values

                # Sanity Checks (Simplified for brevity)
                if global_value is None or any(v is None for v in group_values.values()):
                     print(f"  ERROR: Imputation value is None for {column_name}. Removing from imputation list.")
                     columns_to_impute.remove(column_name)
                     missing_indicator_cols.remove(missing_col_name)
                     del imputation_instructions[column_name]
                     if column_name in global_imputation_values: del global_imputation_values[column_name]
                     if column_name in grouped_imputation_values: del grouped_imputation_values[column_name]

            else:
                # print(f"  '{column_name}' missingness IS NOT significantly associated. Marking as 'not-necessary'.")
                imputation_instructions[column_name] = 'not-necessary'

    # print("--- Imputation Strategy Prepared ---")
    # print(f"Columns to impute: {columns_to_impute}")
    # print(f"Missing indicator columns to create: {missing_indicator_cols}")
    return imputation_instructions, grouped_imputation_values, global_imputation_values, columns_to_impute, missing_indicator_cols


def apply_imputation_strategy(df, imputation_values, columns_to_impute, missing_indicator_cols, target_column=None, mode='grouped'):
    """Applies the prepared imputation strategy to a DataFrame."""
    df_imputed = df.copy()
    # print(f"\n--- Applying Imputation (Mode: {mode}) ---") # Reduce verbosity in pipeline

    # Add missing indicator columns
    # print(f"Adding indicator columns: {missing_indicator_cols}")
    for indicator_col_name in missing_indicator_cols:
        original_col_name = indicator_col_name.replace('_missing', '')
        if original_col_name in df_imputed.columns:
            df_imputed[indicator_col_name] = df_imputed[original_col_name].isna().astype(int)
        # else: print(f"  Warning: Original column '{original_col_name}' not found for indicator '{indicator_col_name}'.")


    # Apply imputation value filling
    # print(f"Imputing columns: {columns_to_impute}")
    for col in columns_to_impute:
        if col not in df_imputed.columns or col not in imputation_values:
             # print(f"  Warning: Column '{col}' or its imputation values not found. Skipping.")
             continue

        nan_count_before = df_imputed[col].isnull().sum()
        if nan_count_before == 0: continue

        if mode == 'grouped':
            if target_column is None or target_column not in df_imputed.columns:
                raise ValueError("Target column must be provided and exist in df for 'grouped' mode.")
            # print(f"  Applying GROUPED imputation for '{col}'...")
            group_vals = imputation_values[col]
            for group, value in group_vals.items():
                 if value is not None:
                     mask = (df_imputed[col].isnull()) & (df_imputed[target_column] == group)
                     df_imputed.loc[mask, col] = value
                 # else: print(f"  Warning: Imputation value is None for column {col}, group {group}. NaNs remain.")

        elif mode == 'global':
            # print(f"  Applying GLOBAL imputation for '{col}'...")
            global_value = imputation_values[col]
            if global_value is not None:
                df_imputed[col].fillna(global_value)
            # else: print(f"  Warning: Global imputation value is None for column {col}. NaNs remain.")
        else:
            raise ValueError("Invalid mode. Choose 'grouped' or 'global'.")

        # nan_count_after = df_imputed[col].isnull().sum()
        # print(f"    '{col}': NaNs imputed: {nan_count_before - nan_count_after} (Remaining NaNs: {nan_count_after})")


    # Optional: Handle 'telephone' - move inside if 'telephone' column exists
    if 'telephone' in df_imputed.columns:
      df_imputed['has_telephone'] = df_imputed['telephone'].notna().astype(int)
      # print("Dropping 'telephone' column.")
      df_imputed.drop(columns=['telephone'], inplace=True, errors='ignore')

    # print("--- Imputation Applied ---")
    return df_imputed

# --- Custom Transformer Class ---
class SmartImputer(BaseEstimator, TransformerMixin):
    def __init__(self, target_column='default'):
        self.target_column = target_column

    def fit(self, X, y=None):
        # y is typically the target Series, used here to ensure target column is available
        # Combine X and y temporarily for prepare function if needed
        # Or ensure target_column is already in X when passed to fit
        if self.target_column not in X.columns:
             if y is not None and isinstance(y, pd.Series) and y.name == self.target_column:
                  X_fit = pd.concat([X, y], axis=1)
             elif y is not None and isinstance(y, pd.Series) and y.name != self.target_column:
                 # Try assigning y with the correct name if lengths match
                 if len(X) == len(y):
                     X_fit = X.copy()
                     X_fit[self.target_column] = y
                 else:
                     raise ValueError(f"Target column '{self.target_column}' not found in X, and provided y has different length or wrong name.")
             else:
                  raise ValueError(f"Target column '{self.target_column}' not found in X and y not provided or suitable.")
        else:
            X_fit = X

        (self.imputation_instructions_, self.grouped_imputation_values_,
         self.global_imputation_values_, self.columns_to_impute_,
         self.missing_indicator_cols_) = prepare_imputation_strategy(X_fit, self.target_column)
        return self

    def transform(self, X):
        # Check if fit has been called
        check_is_fitted(self, ['global_imputation_values_', 'columns_to_impute_', 'missing_indicator_cols_'])
        X_transformed = X.copy()
        # Always use GLOBAL values for transform (inference/test/validation)
        X_transformed = apply_imputation_strategy(
            X_transformed,
            imputation_values=self.global_imputation_values_, # Use learned GLOBAL values
            columns_to_impute=self.columns_to_impute_,
            missing_indicator_cols=self.missing_indicator_cols_,
            target_column=None, # Target not needed/available for global mode
            mode='global' # Explicitly use global mode
        )
        return X_transformed


def convert_object_to_category(df, ordinal=False):
    df = df.copy()
    # Keep track of columns converted to category
    converted_cols = []
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]):
            df[col] = df[col].astype('category')
            converted_cols.append(col)
    # print(f"Converted object columns to category: {converted_cols}") # Optional print

    if ordinal:
      # Ensure columns exist before processing
      columns_to_convert_ordinal = []
      if 'employment_length' in df.columns:
          columns_to_convert_ordinal.append('employment_length')
      if 'residence_history' in df.columns:
           columns_to_convert_ordinal.append('residence_history')

      if not columns_to_convert_ordinal:
           print("Ordinal columns ('employment_length', 'residence_history') not found, skipping ordinal conversion.")
           return df

      def duration_to_months(duration):
          if pd.isna(duration) or not isinstance(duration, str): return np.nan
          if 'month' in duration: return int(duration.split()[0])
          elif 'year' in duration: return int(duration.split()[0]) * 12
          else: return np.nan

      for col in columns_to_convert_ordinal:
          col_name = col.split("_")[0]
          temp_months_col = f'{col_name}_months_temp' # Avoid potential conflicts
          df[temp_months_col] = df[col].astype(str).apply(duration_to_months) # Ensure apply on strings
          unique_months = sorted(df[temp_months_col].dropna().unique())
          # Convert unique months to strings for category levels
          categories = [str(int(m)) if not np.isnan(m) else 'nan' for m in unique_months]
          # Handle potential NaN category explicitly if needed
          if df[temp_months_col].isnull().any() and 'nan' not in categories:
              categories.append('nan') # Add 'nan' if missing values exist and aren't a level

          # Convert the temp months column to string, handling NaNs
          series_to_convert = df[temp_months_col].apply(lambda x: str(int(x)) if pd.notna(x) else 'nan')

          # Create the ordered categorical column
          df[f'{col_name}_ordinal'] = pd.Categorical(
              series_to_convert,
              categories=categories,
              ordered=True
          )
          # print(f"Created ordinal column: {col_name}_ordinal") # Optional print
          # Drop original and temporary columns
          df.drop(columns=[col, temp_months_col], inplace=True, errors='ignore')

    return df

def regroup_installment_plan(df: pd.DataFrame, column_name: str = 'installment_plan') -> pd.DataFrame:
    """
    Regroups the 'installment_plan' column based on observed default rates.

    Combines 'bank' and 'stores' into a single 'Bank/Stores Plan' category,
    leaving 'none' as its own category. Creates a new column 'installment_plan_grouped'.

    Args:
        df (pd.DataFrame): The input DataFrame containing the installment_plan column.
        column_name (str): The name of the column to regroup. Defaults to 'installment_plan'.

    Returns:
        pd.DataFrame: The DataFrame with the new 'installment_plan_grouped' column added.
                      The original column is kept for reference but can be dropped later.
    """
    df_copy = df.copy()

    if column_name not in df_copy.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")

    # Define the mapping for regrouping
    # Categories 'bank' and 'stores' have similar high default rates (~40-41%)
    # Category 'none' has a distinct lower default rate (~27.5%)
    mapping = {
        'bank': 'Bank/Stores Plan',
        'stores': 'Bank/Stores Plan',
        'none': 'None'
    }

    # Create the new grouped column
    new_col_name = f"{column_name}_grouped"
    df_copy[new_col_name] = df_copy[column_name].map(mapping)

    # Convert the new column to category dtype for consistency
    df_copy[new_col_name] = df_copy[new_col_name].astype('category')

    return df_copy


def engineer_numerical_features(df):
  df_copy = df.copy()
  # Ensure source columns exist and handle potential division by zero or NaNs
  if "amount" in df_copy and "months_loan_duration" in df_copy:
    df_copy["loan_per_month"] = df_copy["amount"] / df_copy["months_loan_duration"].replace(0, 1e-6) # Avoid zero div
  if "amount" in df_copy and "checking_balance" in df_copy:
    df_copy["loan_to_checking_ratio"] = df_copy["amount"] / (df_copy["checking_balance"].fillna(0) + 1e-6) # Fill NaNs before division
  if "amount" in df_copy and "savings_balance" in df_copy:
    df_copy["loan_to_savings_ratio"] = df_copy["amount"] / (df_copy["savings_balance"].fillna(0) + 1e-6)
  if "amount" in df_copy and "checking_balance" in df_copy and "savings_balance" in df_copy:
    df_copy["loan_to_total_ratio"] = df_copy["amount"] / (df_copy["checking_balance"].fillna(0) + df_copy["savings_balance"].fillna(0) + 1e-6)
  # print("Engineered numerical features.") # Optional print
  return df_copy


def engineer_categorical_features(df):
  df_copy = df.copy() # Work on a copy

  # Foreign Worker-Job Interactions (ensure source columns exist)
  if 'foreign_worker' in df_copy and 'job' in df_copy:
      df_copy['is_foreign_worker'] = df_copy['foreign_worker'].map({'yes': 1, 'no': 0}).fillna(0).astype(int)
      job_one_hot = pd.get_dummies(df_copy['job'], prefix='job', dtype=int)
      interaction_features = job_one_hot.multiply(df_copy['is_foreign_worker'], axis=0)
      interaction_features.columns = [f'{col}_AND_foreign' for col in interaction_features.columns]
      # Only concat columns that aren't already there
      cols_to_add = [col for col in interaction_features.columns if col not in df_copy.columns]
      df_copy = pd.concat([df_copy, interaction_features[cols_to_add]], axis=1)
  # else: print("Skipping foreign worker interaction - required columns missing.")

  # Job and Housing Combinations
  if 'job' in df_copy and 'housing' in df_copy:
      df_copy['job_housing_interaction'] = df_copy['job'].astype(str) + '_' + df_copy['housing'].astype(str)
      df_copy['job_housing_interaction'] = df_copy['job_housing_interaction'].astype('category')
  # else: print("Skipping job-housing interaction - required columns missing.")

  # Personal Status and Housing
  if 'personal_status' in df_copy and 'housing' in df_copy:
      df_copy['personal_status_housing_interaction'] = df_copy['personal_status'].astype(str) + '_' + df_copy['housing'].astype(str)
      df_copy['personal_status_housing_interaction'] = df_copy['personal_status_housing_interaction'].astype('category')
  # else: print("Skipping personal_status-housing interaction - required columns missing.")

  # Credit History and Loan Purpose interaction
  if 'credit_history' in df_copy and 'purpose' in df_copy:
      df_copy['credit_history_purpose_interaction'] = df_copy['credit_history'].astype(str) + '_' + df_copy['purpose'].astype(str)
      df_copy['credit_history_purpose_interaction'] = df_copy['credit_history_purpose_interaction'].astype('category')
  # else: print("Skipping credit_history-purpose interaction - required columns missing.")

  # print("Engineered categorical features.") # Optional print
  return df_copy

# Wrap functions that don't require fitting state
# Set ordinal=True if you want the ordinal conversion logic to run
convert_cats_transformer = FunctionTransformer(convert_object_to_category, kw_args={'ordinal': True}, validate=False)
regroup_installment_plan_transformer = FunctionTransformer(regroup_installment_plan, validate=False)
engineer_num_transformer = FunctionTransformer(engineer_numerical_features, validate=False)

best_params = OrderedDict([
    ('colsample_bytree', 1.0),
    ('learning_rate', 0.16455239155781587),
    ('max_depth', 9),
    ('subsample', 1.0)
])

xgb_model = xgb.XGBClassifier(
    **best_params,
    n_estimators=100,
    objective='binary:logistic',
    use_label_encoder=False,
    eval_metric='logloss',
    enable_categorical=True, # Set to False because OHE is used
    random_state=42
    )

# --- Build the Full Pipeline ---
training_pipeline = Pipeline(steps=[
    ('imputer', SmartImputer(target_column='default')),
    ('convert_types', convert_cats_transformer), # Convert objects, create ordinals
    ('regroup_installment_plan', regroup_installment_plan_transformer), # Regroup installment plan
    ('engineer_numerical', engineer_num_transformer), # Create numerical features
    # ('engineer_categorical', engineer_cat_transformer), # Create categorical features
    ('classifier', xgb_model) # Add the classifier
    # Use xgb_model_cat if using enable_categorical=True and adjusted preprocessor_standard
])

print("\nFull Training Pipeline Created:")
print(training_pipeline)

if __name__ == "__main__":
    # Example usage (assuming you have a DataFrame `df`):
    target = 'default'  # Replace with your target column name
    df = pd.read_csv('GermanCredit.csv')  # Load your data
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame.")
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')  # Drop index column if exists

    y_train = df[target]
    X_train_full_features = df.drop(columns=[target])

    # Fit the pipeline on the training data
    training_pipeline.fit(X_train_full_features, y_train)
    # Save the pipeline to a file
    # --- Save the Fitted Pipeline ---
    pipeline_filename = 'full_credit_pipeline_new.joblib'
    pipeline_path = f"{pipeline_filename}"
    joblib.dump(training_pipeline, pipeline_path)
    print(f"Fitted pipeline saved to {pipeline_filename}")
