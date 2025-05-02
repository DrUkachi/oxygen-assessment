import pandas as pd
import json
from datetime import datetime

class CreditScoringFeatureExtractor:
    def __init__(self, json_file_path):
        """
        Initialize the CreditScoringFeatureExtractor with a JSON file and load it into a DataFrame.

        :param json_file_path: Path to the JSON file.
        """
        with open(json_file_path, 'r') as file:
            data = json.load(file)

        # Normalize the main JSON data
        self.df = pd.json_normalize(data)

    def calculate_credit_health_ratio(self):
        """
        Calculate the total number of good and bad accounts, then compute the Credit Health Ratio.
        """
        self.df['Total_Good_Accounts'] = (
            self.df['data.consumerfullcredit.accountrating.noofotheraccountsgood'].astype(int) +
            self.df['data.consumerfullcredit.accountrating.noofretailaccountsgood'].astype(int) +
            self.df['data.consumerfullcredit.accountrating.noofcreditcardaccountsgood'].astype(int) +
            self.df['data.consumerfullcredit.accountrating.noofpersonalloanaccountsgood'].astype(int)
        )

        self.df['Total_Bad_Accounts'] = (
            self.df['data.consumerfullcredit.accountrating.noofotheraccountsbad'].astype(int) +
            self.df['data.consumerfullcredit.accountrating.noofretailaccountsbad'].astype(int) +
            self.df['data.consumerfullcredit.accountrating.noofcreditcardaccountsbad'].astype(int) +
            self.df['data.consumerfullcredit.accountrating.noofpersonalloanaccountsbad'].astype(int)
        )

        self.df['Credit_Health_Ratio'] = (
            self.df['Total_Good_Accounts'] /
            (self.df['Total_Good_Accounts'] + self.df['Total_Bad_Accounts'])
        )

        return self.df[['Total_Good_Accounts', 'Total_Bad_Accounts', 'Credit_Health_Ratio']]

    def extract_gurantor_count(self):
        """
        Rename and extract the guarantor count as a new feature.
        """
        self.df.rename(columns={
            'data.consumerfullcredit.guarantorcount.guarantorssecured': 'guarantor_count'},
            inplace=True)
        return self.df[['guarantor_count']]

    def calculate_recency_features(self):
        """
        Calculate recency features based on the phone number update dates.

        This method computes the number of days since the last update of home and mobile phone numbers
        from the telephone history. If the necessary data is missing, it will handle the errors gracefully.

        Returns:
            pd.DataFrame: A DataFrame containing the days since home and mobile updates.
        """
        today = datetime.today()

        def calculate_days_since_updates(telephone_history):
            if telephone_history:  # Ensure there's telephone history data
                history_df = pd.json_normalize(telephone_history)

                # Check if the necessary columns exist
                if 'homenoupdatedondate' in history_df.columns and 'mobilenoupdatedondate' in history_df.columns:
                    # Convert date columns to datetime format
                    history_df['homenoupdatedondate'] = pd.to_datetime(history_df['homenoupdatedondate'], format='%d/%m/%Y', errors='coerce')
                    history_df['mobilenoupdatedondate'] = pd.to_datetime(history_df['mobilenoupdatedondate'], format='%d/%m/%Y', errors='coerce')

                    # Calculate days since updates
                    days_since_home = (today - history_df['homenoupdatedondate']).dt.days.min() if not history_df['homenoupdatedondate'].isnull().all() else None
                    days_since_mobile = (today - history_df['mobilenoupdatedondate']).dt.days.min() if not history_df['mobilenoupdatedondate'].isnull().all() else None

                    return pd.Series([days_since_home, days_since_mobile])

            # Return None if telephone history is missing or incomplete
            return pd.Series([None, None])

        # Apply the function across the DataFrame
        self.df[['Days_Since_Home_Update', 'Days_Since_Mobile_Update']] = self.df['data.consumerfullcredit.telephonehistory'].apply(calculate_days_since_updates)

        return self.df[['Days_Since_Home_Update', 'Days_Since_Mobile_Update']]


    def calculate_phone_number_presence(self):
        """
        Calculate binary features indicating the presence of phone numbers.

        Returns:
            pd.DataFrame: A DataFrame containing binary features indicating the presence of home, mobile, and work phone numbers.
        """
        # Initialize new columns
        self.df['Has_Home_Phone'] = None
        self.df['Has_Mobile_Phone'] = None
        self.df['Has_Work_Phone'] = None

        # Iterate over rows to calculate presence of phone numbers per row
        for idx, row in self.df.iterrows():
            telephone_history = row.get('data.consumerfullcredit.telephonehistory', [])
            if telephone_history:  # Ensure there's telephone history data
                history_df = pd.json_normalize(telephone_history)

                # Calculate presence of phone numbers with key checks
                has_home_phone = 'hometelephonenumber' in history_df.columns and history_df['hometelephonenumber'].apply(lambda x: x != 'XXX').any()
                has_mobile_phone = 'mobiletelephonenumber' in history_df.columns and history_df['mobiletelephonenumber'].apply(lambda x: x != 'XXX').any()
                has_work_phone = 'worktelephonenumber' in history_df.columns and history_df['worktelephonenumber'].apply(lambda x: pd.notnull(x) and x != 'XXX').any()

                # Assign to the main DataFrame
                self.df.at[idx, 'Has_Home_Phone'] = has_home_phone
                self.df.at[idx, 'Has_Mobile_Phone'] = has_mobile_phone
                self.df.at[idx, 'Has_Work_Phone'] = has_work_phone

        return self.df[['Has_Home_Phone', 'Has_Mobile_Phone', 'Has_Work_Phone']].astype(bool)

    def calculate_duration_between_updates(self):
        """
        Calculate the duration (in days) between consecutive home phone updates and mobile phone updates.

        This will provide a measure of how frequent home or mobile phone changes are made.

        Returns:
            pd.DataFrame: A DataFrame containing the average duration between home and mobile phone updates.
        """
        # Initialize the new columns for home and mobile updates
        self.df['Home_Phone_Update_Frequency'] = None
        self.df['Mobile_Phone_Update_Frequency'] = None

        # Iterate over rows to calculate update frequencies per row
        for idx, row in self.df.iterrows():
            telephone_history = row.get('data.consumerfullcredit.telephonehistory', [])
            if telephone_history:  # Ensure there's telephone history data
                history_df = pd.json_normalize(telephone_history)

                # Convert date columns to datetime format with key checks
                if 'homenoupdatedondate' in history_df.columns:
                    history_df['homenoupdatedondate'] = pd.to_datetime(history_df['homenoupdatedondate'], format='%d/%m/%Y', errors='coerce')
                    # Sort the dates for home updates to calculate consecutive differences
                    home_durations = history_df['homenoupdatedondate'].diff().dt.days.dropna()
                    home_update_freq = home_durations.mean() if not home_durations.empty else None
                    self.df.at[idx, 'Home_Phone_Update_Frequency'] = home_update_freq

                if 'mobilenoupdatedondate' in history_df.columns:
                    history_df['mobilenoupdatedondate'] = pd.to_datetime(history_df['mobilenoupdatedondate'], format='%d/%m/%Y', errors='coerce')
                    # Sort the dates for mobile updates to calculate consecutive differences
                    mobile_durations = history_df['mobilenoupdatedondate'].diff().dt.days.dropna()
                    mobile_update_freq = mobile_durations.mean() if not mobile_durations.empty else None
                    self.df.at[idx, 'Mobile_Phone_Update_Frequency'] = mobile_update_freq

        return self.df[['Home_Phone_Update_Frequency', 'Mobile_Phone_Update_Frequency']].astype(float)


    def extract_employment_features(self):
        """
        Extract useful employment history features such as most recent occupation,
        number of unique occupations, and job stability metrics using apply().
        """
        def process_employment_history(employment_history):
            if not employment_history:
                return pd.Series({
                    'Latest_Occupation': None,
                    'Unique_Occupations': None,
                    'Unique_Employers': None,
                    'Employment_Update_Recency': None,
                    'Is_Public_Servant': 0
                })

            history_df = pd.json_normalize(employment_history)
            # Ensure the 'updateondate' column exists before processing
            if 'updateondate' in history_df.columns:
                history_df['updateondate'] = pd.to_datetime(history_df['updateondate'], format='%d/%m/%Y', errors='coerce')

                # Most recent occupation and employer details
                latest_update = history_df.sort_values(by='updateondate', ascending=False).iloc[0]
                latest_occupation = latest_update.get('occupation')
                latest_employer = latest_update.get('employerdetail')
            else:
                latest_occupation = None
                latest_employer = None

            # Count unique occupations and employers safely
            unique_occupations = history_df['occupation'].nunique() if 'occupation' in history_df.columns else 0
            unique_employers = history_df['employerdetail'].nunique() if 'employerdetail' in history_df.columns else 0


            # Employment recency (how recent is the latest update)
            if 'updateondate' in history_df.columns and pd.notnull(latest_update.get('updateondate')):
                employment_update_recency = (datetime.today() - latest_update['updateondate']).days
            else:
                employment_update_recency = None

            # Indicator if ever a public servant
            is_public_servant = int('PUBLIC SERVANTS' in history_df['occupation'].values) if 'occupation' in history_df.columns else 0

            return pd.Series({
                'Latest_Occupation': latest_occupation,
                'Latest_Employer': latest_employer,
                'Unique_Occupations': int(unique_occupations),
                'Unique_Employers': int(unique_employers),
                'Employment_Update_Recency': employment_update_recency,
                'Is_Public_Servant': bool(is_public_servant)
                })

        # Apply the function row-wise
        employment_features = self.df['data.consumerfullcredit.employmenthistory'].apply(process_employment_history)

        # Concatenate the new features back to the main DataFrame
        self.df = pd.concat([self.df, employment_features], axis=1)

        return self.df[['Latest_Occupation', 'Unique_Occupations', 'Unique_Employers', 'Employment_Update_Recency', 'Is_Public_Servant']]

    def extract_enquiry_features(self):
        """
        Extract useful enquiry history features such as the total number of enquiries,
        enquiry recency, and the distribution of enquiry reasons using apply().
        """
        def process_enquiry_history(enquiry_history):
            if not enquiry_history:
                return pd.Series({
                    'Total_Enquiries': 0,
                    'Recent_Enquiry_Days': None,
                    'Credit_Scoring_Enquiries': 0,
                    'Consent_Enquiries': 0,
                    'Application_Enquiries': 0,
                    'Unique_Subscribers': 0,
                    'Avg_Time_Between_Enquiries': None
                })

            history_df = pd.json_normalize(enquiry_history)
            history_df['daterequested'] = pd.to_datetime(history_df['daterequested'], format='%d/%m/%Y %H:%M:%S', errors='coerce')

            # Total number of enquiries
            total_enquiries = len(history_df)

            # Most recent enquiry (in days)
            recent_enquiry_days = (datetime.today() - history_df['daterequested'].max()).days if pd.notnull(history_df['daterequested'].max()) else None

            # Enquiries by reason
            credit_scoring_enquiries = history_df[history_df['enquiryreason'].str.contains('Credit scoring', na=False)].shape[0]
            consent_enquiries = history_df[history_df['enquiryreason'].str.contains('consent', na=False)].shape[0]
            application_enquiries = history_df[history_df['enquiryreason'].str.contains('application', na=False)].shape[0]

            # Unique subscribers
            unique_subscribers = history_df['subscribername'].nunique()

            # Calculate average time between enquiries
            history_df = history_df.sort_values(by='daterequested')
            time_diffs = history_df['daterequested'].diff().dt.days.dropna()
            avg_time_between_enquiries = time_diffs.mean() if not time_diffs.empty else None

            return pd.Series({
                'Total_Enquiries': int(total_enquiries),
                'Recent_Enquiry_Days': int(recent_enquiry_days),
                'Credit_Scoring_Enquiries': int(credit_scoring_enquiries),
                'Consent_Enquiries': int(consent_enquiries),
                'Application_Enquiries': int(application_enquiries),
                'Unique_Subscribers': int(unique_subscribers),
                'Avg_Time_Between_Enquiries': float(avg_time_between_enquiries)
            })

        # Apply the function row-wise
        enquiry_features = self.df['data.consumerfullcredit.enquiryhistorytop'].apply(process_enquiry_history)

        # Concatenate the new features back to the main DataFrame
        self.df = pd.concat([self.df, enquiry_features], axis=1)

        return self.df[['Total_Enquiries', 'Recent_Enquiry_Days', 'Credit_Scoring_Enquiries', 'Consent_Enquiries', 'Application_Enquiries', 'Unique_Subscribers', 'Avg_Time_Between_Enquiries']]


    def calculate_delinquency_features(self):
        """
        Calculate various delinquency features based on the delinquency information.
        """
        # Ensure relevant columns are in the correct type
        self.df['monthsinarrears'] = pd.to_numeric(self.df['data.consumerfullcredit.deliquencyinformation.monthsinarrears'], errors='coerce').fillna(0)
        self.df['periodnum'] = pd.to_datetime(self.df['data.consumerfullcredit.deliquencyinformation.periodnum'], format='%Y%m%d', errors='coerce')

        # Calculate account age in months (assuming the current date)
        current_date = pd.to_datetime('today')
        self.df['Account_Age'] = ((current_date - self.df['periodnum']).dt.days // 30).fillna(0)

        # Create a binary feature for recent delinquency (e.g., in the last 6 months)
        six_months_ago = current_date - pd.DateOffset(months=6)
        self.df['Is_Recently_Delinquent'] = (self.df['periodnum'] >= six_months_ago).astype(bool)

        self.df['deliquent_subscriber_name'] = self.df['data.consumerfullcredit.deliquencyinformation.subscribername']


        return self.df[['monthsinarrears',
                    'periodnum',
                    'deliquent_subscriber_name', # To make it different for other information relating to subscriber_name
                    'Account_Age',
                    'Is_Recently_Delinquent']]

    def extract_credit_agreement_features(self):
        """
        Extract useful features from the credit agreement summary,
        such as total accounts, total open accounts, total closed accounts,
        average balance, and performance status.
        """
        def process_credit_agreements(agreement_summary):
            if not agreement_summary:
                return pd.Series({
                    'Total_Accounts': 0,
                    'Total_Open_Accounts': 0,
                    'Total_Closed_Accounts': 0,
                    'Average_Balance': 0.0,
                    'Performing_Accounts': 0,
                    'Non_Performing_Accounts': 0
                })

            agreements_df = pd.json_normalize(agreement_summary)
            agreements_df['currentbalanceamt'] = agreements_df['currentbalanceamt'].replace('[\$,]', '', regex=True).astype(float)

            # Total accounts
            total_accounts = len(agreements_df)

            # Open and closed accounts
            total_open_accounts = int((agreements_df['accountstatus'] == 'Open').sum())
            total_closed_accounts = int((agreements_df['accountstatus'] == 'Closed').sum())

            # Average balance
            average_balance = agreements_df['currentbalanceamt'].mean() if total_accounts > 0 else 0.0

            # Performing and non-performing accounts
            performing_accounts = int((agreements_df['performancestatus'] == 'Performing').sum())
            non_performing_accounts = total_accounts - performing_accounts

            # Total Loan Duration
            total_loan_duration = agreements_df['loanduration'].astype(float).sum()

            # Count of unique subscriber names
            subscriber_name_count = agreements_df['subscribername'].nunique()

            return pd.Series({
                'Total_Accounts': total_accounts,
                'Total_Open_Accounts': total_open_accounts,
                'Total_Closed_Accounts': total_closed_accounts,
                'Average_Balance': average_balance,
                'Performing_Accounts': performing_accounts,
                'Non_Performing_Accounts': non_performing_accounts,
                'Total_Loan_Duration':total_loan_duration,
                'Subcriber_Name_Count':int(subscriber_name_count)

            })

        # Apply the function row-wise
        credit_agreement_features = self.df['data.consumerfullcredit.creditagreementsummary'].apply(process_credit_agreements)

        # Concatenate the new features back to the main DataFrame
        self.df = pd.concat([self.df, credit_agreement_features], axis=1)

        return self.df[['Total_Accounts',
                        'Total_Open_Accounts',
                        'Total_Closed_Accounts',
                        'Average_Balance',
                        'Performing_Accounts',
                        'Non_Performing_Accounts',
                        'Total_Loan_Duration',
                        'Subcriber_Name_Count']]

    def extract_personal_details_features(self):
        """
        Extract useful personal details features such as age, gender,
        number of dependants, residential stability, and more.
        """
        def process_personal_details(details):
            if details.empty:
                return pd.Series({

                    'Age': None,
                    'Gender': None,
                    'Number_of_Dependents': 0,
                    'Residential_Stability': None,
                    'has_Contact': False,
                    'Nationality': None,
                    'Employment_Status': None,
                    'Property_Owned': None
                })

            # Safely extract fields from the details using direct access

            birthdate = pd.to_datetime(details.get('data.consumerfullcredit.personaldetailssummary.birthdate'), format='%d/%m/%Y', errors='coerce')
            age = (datetime.today() - birthdate).days // 365 if pd.notnull(birthdate) else None
            gender = details.get('data.consumerfullcredit.personaldetailssummary.gender', None)
            dependants = int(details.get('data.consumerfullcredit.personaldetailssummary.dependants', 0))
            residential_stability = all(details.get(key) is not None for key in [
                'data.consumerfullcredit.personaldetailssummary.residentialaddress1',
                'data.consumerfullcredit.personaldetailssummary.residentialaddress2'
            ])
            has_contact = details.get('data.consumerfullcredit.personaldetailssummary.cellularno') is not None
            nationality = details.get('data.consumerfullcredit.personaldetailssummary.nationality', None)
            employment_status = details.get('data.consumerfullcredit.personaldetailssummary.employerdetail', None)
            property_owned = details.get('data.consumerfullcredit.personaldetailssummary.propertyownedtype', None)

            return pd.Series({
                'Age': age,
                'Gender': gender,
                'Number_of_Dependents': dependants,
                'Residential_Stability': residential_stability,
                'has_Contact': has_contact,
                'Nationality': nationality,
                'Employment_Status': employment_status,
                'Property_Owned': property_owned
            })

        # Apply the function row-wise
        personal_features = self.df.apply(process_personal_details, axis=1)

        # Concatenate the new features back to the main DataFrame
        self.df = pd.concat([self.df, personal_features], axis=1)

        return self.df[['application_id',  # Include application_id in the output
                        'Age',
                        'Gender',
                        'Number_of_Dependents',
                        'Residential_Stability',
                        'has_Contact',
                        'Nationality',
                        'Employment_Status',
                        'Property_Owned']]

    def compile_engineered_features(self):
        """
        Compile all engineered features into a single DataFrame and return it.

        This method calls all feature extraction methods defined in the class,
        compiles their results into a single DataFrame, and returns it.

        Usage:
            extractor = CreditScoringFeatureExtractor('path/to/your/json/file.json')
            compiled_features_df = extractor.compile_engineered_features()

        Returns:
            pd.DataFrame: A DataFrame containing all engineered features.
        """
        features = {}

        # Add all the features from different methods
        features['Credit_Health_Ratio'] = self.calculate_credit_health_ratio()
        features['Guarantor_Count'] = self.extract_gurantor_count()
        features['Recency_Features'] = self.calculate_recency_features()
        features['Phone_Number_Presence'] = self.calculate_phone_number_presence()
        features['Update_Frequency'] = self.calculate_duration_between_updates()
        features['Employment_Features'] = self.extract_employment_features()
        features['Enquiry_Features'] = self.extract_enquiry_features()
        features['Delinquency_Features'] = self.calculate_delinquency_features()
        features['Credit_Agreement_Features'] = self.extract_credit_agreement_features()
        features['Personal_Details_Features'] = self.extract_personal_details_features()


        # Compile all features into a single DataFrame
        compiled_df = pd.concat(features.values(), axis=1)

        # Derived Features - From original features
        compiled_df['open_accounts_ratio'] = compiled_df['Total_Open_Accounts'] / compiled_df['Total_Accounts']
        compiled_df['closed_accounts_ratio'] = compiled_df['Total_Closed_Accounts'] / compiled_df['Total_Accounts']
        compiled_df['performing_accounts_ratio'] = compiled_df['Performing_Accounts'] / compiled_df['Total_Accounts']

        return compiled_df
    
if __name__ == "__main__":
    # Example usage
    JSON_FILE_PATH = 'path/to/your/json/file.json'  # Replace with your JSON file path
    extractor = CreditScoringFeatureExtractor(JSON_FILE_PATH)
    features_df = extractor.compile_engineered_features()
    print(features_df.head())
    # Save the features to a CSV file
    features_df.to_csv('engineered_features.csv', index=False)
    print("Engineered features saved to 'engineered_features.csv'")
