
# ML Features with Importance and Categories

1. **application_id**
   - **Importance:** Not used as a direct model input feature, but crucial for tracking individual predictions and linking model outputs back to specific applications for analysis, monitoring, and potential retraining.
   - **Category:** None
   - **Data Type:** Integer

2. **Total_Good_Accounts**
   - **Importance:** Predictive of lower default risk; a higher count signals responsible credit management and a stronger repayment history, improving model accuracy in identifying creditworthy applicants.
   - **Category:** Account Rating
   - **Data Type:** Integer

3. **Total_Bad_Accounts**
   - **Importance:** Predictive of higher default risk; a higher count indicates past difficulties in managing debt, serving as a strong negative signal for the risk model.
   - **Category:** Account Rating
   - **Data Type:** Integer

4. **Credit_Health_Ratio**
   - **Importance:** Offers a composite view of credit management; a higher ratio is predictive of lower risk, potentially capturing repayment consistency better than raw counts alone.
   - **Category:** Account Rating
   - **Data Type:** Float

5. **guarantor_count**
   - **Importance:** May signal higher inherent risk in the primary applicant (requiring guarantors), potentially refining risk stratification, although its predictive power might depend on lender policies.
   - **Category:** Guarantor Count
   - **Data Type:** Integer

6. **Days_Since_Home_Update**
   - **Importance:** Longer periods since an update could indicate lower contact data reliability or potential instability, possibly correlating with increased risk. Recency might signal stability.
   - **Category:** Telephone History
   - **Data Type:** Integer

7. **Days_Since_Mobile_Update**
   - **Importance:** Similar to home phone updates, longer periods might suggest lower contactability or instability, potentially correlating with higher credit risk.
   - **Category:** Telephone History
   - **Data Type:** Integer

8. **Has_Home_Phone**
   - **Importance:** Historically a proxy for residential stability; its presence might correlate with slightly lower default risk, though its predictive power may be diminishing.
   - **Category:** Telephone History
   - **Data Type:** Boolean

9. **Has_Mobile_Phone**
   - **Importance:** Essential for contactability; its absence might be a weak risk signal, while its presence is largely expected but necessary for basic communication.
   - **Category:** Telephone History
   - **Data Type:** Boolean

10. **Has_Work_Phone**
    - **Importance:** Can act as a proxy for employment stability; its presence may correlate with lower default risk.
    - **Category:** Telephone History
    - **Data Type:** Boolean

11. **Home_Phone_Update_Frequency**
    - **Importance:** A high frequency of updates could signal residential instability, potentially increasing the predicted default risk.
    - **Category:** Telephone History
    - **Data Type:** Integer

12. **Mobile_Phone_Update_Frequency**
    - **Importance:** Frequent updates might indicate general instability (address, job changes), potentially correlating with higher credit risk.
    - **Category:** Telephone History
    - **Data Type:** Integer

13. **Latest_Occupation**
    - **Importance:** Different occupations carry varying levels of income stability and risk; incorporating this allows the model to learn industry/job-specific risk patterns. (Requires encoding).
    - **Category:** Employment History
    - **Data Type:** String

14. **Unique_Occupations**
    - **Importance:** A high number might indicate job instability (higher risk) or diverse experience (neutral/lower risk); its predictive value needs to be learned by the model.
    - **Category:** Employment History
    - **Data Type:** Integer

15. **Unique_Employers**
    - **Importance:** Frequent employer changes (high count) can be predictive of job instability and potentially higher credit risk.
    - **Category:** Employment History
    - **Data Type:** Integer

16. **Employment_Update_Recency**
    - **Importance:** More recent updates (lower value) suggest current employment information, potentially signaling lower immediate risk compared to stale data.
    - **Category:** Employment History
    - **Data Type:** Float

17. **Is_Public_Servant**
    - **Importance:** Often associated with high job security, making it predictive of lower default risk.
    - **Category:** Employment History
    - **Data Type:** Boolean

18. **Total_Enquiries**
    - **Importance:** A high number of recent credit inquiries is a strong predictor of increased credit risk, signaling active credit seeking which can correlate with financial distress.
    - **Category:** Enquiry Details
    - **Data Type:** Integer

19. **Recent_Enquiry_Days**
    - **Importance:** Very recent inquiries (low value) are more predictive of imminent default risk than older inquiries.
    - **Category:** Enquiry Details
    - **Data Type:** Integer

20. **Credit_Scoring_Enquiries**
    - **Importance:** Multiple inquiries specifically for credit scoring can indicate a borrower is comparing offers or facing rejections, potentially signaling higher risk.
    - **Category:** Enquiry Details
    - **Data Type:** Integer

21. **Consent_Enquiries**
    - **Importance:** Its predictive value might be nuanced; while expected, a lack of consent inquiries where expected could be a flag, though less direct than other enquiry types.
    - **Category:** Enquiry Details
    - **Data Type:** Integer

22. **Application_Enquiries**
    - **Importance:** A high number of application inquiries, especially recent ones, strongly suggests active credit seeking and correlates with higher default risk.
    - **Category:** Enquiry Details
    - **Data Type:** Integer

23. **Unique_Subscribers**
    - **Importance:** Reflects the breadth of an applicant's credit relationships; a high number might indicate complex credit usage (potentially higher risk) or established history, requiring the model to learn the pattern.
    - **Category:** Enquiry Details
    - **Data Type:** Integer

24. **Avg_Time_Between_Enquiries**
    - **Importance:** Shorter average times between inquiries are predictive of higher risk, indicating frequent or urgent credit seeking behavior.
    - **Category:** Enquiry Details
    - **Data Type:** Float

25. **monthsinarrears**
    - **Importance:** A direct measure of current or recent delinquency severity; any value greater than zero is a strong predictor of significantly higher default risk.
    - **Category:** Delinquency Information
    - **Data Type:** Integer

26. **periodnum**
    - **Importance:** If indicating the recency or duration of delinquency periods, it's highly predictive of risk. More recent or longer periods signal higher risk. Needs precise definition.
    - **Category:** Delinquency Information
    - **Data Type:** Datetime

27. **deliquent_subscriber_name**
    - **Importance:** The *type* of subscriber reporting delinquency (e.g., payday lender vs. bank) can carry different risk implications; allows the model to learn lender-specific risk patterns associated with defaults. (Requires encoding).
    - **Category:** Delinquency Information
    - **Data Type:** String

28. **Account_Age**
    - **Importance:** Longer account history (higher age) is generally predictive of lower risk, indicating more experience managing credit.
    - **Category:** Credit Account Summary
    - **Data Type:** Integer

29. **Is_Recently_Delinquent**
    - **Importance:** A powerful binary flag; being recently delinquent is a very strong predictor of future default risk.
    - **Category:** Delinquency Information
    - **Data Type:** Boolean

30. **Total_Accounts**
    - **Importance:** Provides context for other credit metrics; its direct impact on risk is often less significant than utilization or delinquency measures but necessary for calculating ratios.
    - **Category:** Credit Account Summary
    - **Data Type:** Integer

31. **Total_Open_Accounts**
    - **Importance:** Relates to current credit exposure and utilization; a high number relative to total accounts might signal higher risk if balances are high.
    - **Category:** Credit Account Summary
    - **Data Type:** Integer

32. **Total_Closed_Accounts**
    - **Importance:** Reflects past credit activity; a history of successfully closed accounts (paid off) can be predictive of lower risk when considered with other factors.

    - **Category:** Credit Account Summary
    - **Data Type:** Integer

33. **Average_Balance** *(Needs context of account type)*
    - **Importance:** Highly dependent on account type (e.g., high credit card balance vs. high mortgage balance); its value in risk scoring is refined when combined with account type or used in utilization ratios. High revolving debt balances are predictive of higher risk.
    - **Category:** Credit Account Summary
    - **Data Type:** Float

34. **Performing_Accounts**
    - **Importance:** Similar to 'Total Good Accounts', a higher number relative to total accounts is predictive of lower risk.
    - **Category:** Credit Account Summary
    - **Data Type:** Integer

35. **Non_Performing_Accounts**
    - **Importance:** Similar to 'Total Bad Accounts', a higher number is predictive of significantly higher risk.
    - **Category:** Credit Account Summary
    - **Data Type:** Integer

36. **Total_Loan_Duration** *(Needs more context)*
    - **Importance:** If representing average or total duration of *existing* loans, longer durations might indicate higher lifetime exposure but also established relationships; predictive value depends on interaction with other factors.
    - **Category:** Credit Account Summary
    - **Data Type:** Integer

37. **Subcriber_Name_Count** *(Needs more context)*
    - **Importance:** Similar to 'Unique Subscribers', reflects the diversity of credit relationships; its link to risk needs to be learned by the model.
    - **Category:** Credit Account Summary
    - **Data Type:** Integer

38. **Age**
    - **Importance:** Younger applicants often have shorter credit histories and less stable income, correlating with higher risk; older applicants generally show lower risk.
    - **Category:** Personal Details Summary
    - **Data Type:** Integer

39. **Gender**
    - **Importance:** While potentially correlated with risk in historical data, using it directly can introduce bias. Its primary relevance is for fairness analysis to ensure the model doesn't create disparate outcomes based on gender.
    - **Category:** Personal Details Summary
    - **Data Type:** String

40. **Number_of_Dependents**
    - **Importance:** A higher number can indicate greater financial obligations, potentially correlating with increased default risk, especially if income is limited.
    - **Category:** Personal Details Summary
    - **Data Type:** Integer

41. **Residential_Stability**
    - **Importance:** A direct indicator of stability; higher stability (True) is predictive of lower default risk.
    - **Category:** Personal Details Summary
    - **Data Type:** Boolean

42. **has_Contact**
    - **Importance:** Lack of contact information can hinder communication and collections, potentially correlating slightly with higher risk or application incompleteness.
    - **Category:** Personal Details Summary
    - **Data Type:** Boolean

43. **Nationality**
    - **Importance:** Similar to Gender, using Nationality directly as a predictor carries significant bias risk. Its main role is in fairness evaluation, though the model might uncover historical correlations.
    - **Category:** Personal Details Summary
    - **Data Type:** String

44. **Employment_Status**
    - **Importance:** Certain statuses (e.g., 'Unemployed', 'Temporary') are highly predictive of increased default risk compared to 'Employed' or 'Self-Employed' (which has its own risk profile). (Requires encoding).
    - **Category:** Personal Details Summary
    - **Data Type:** String

45. **Property_Owned**
    - **Importance:** Owning property (especially outright) is often a strong indicator of financial stability and collateral, making it predictive of lower default risk compared to renting. (Requires encoding).
    - **Category:** Personal Details Summary
    - **Data Type:** String

46. **open_accounts_ratio**
   - **Importance:** Provides insight into active credit management. A very high ratio might indicate newer credit history or taking on much debt recently (higher risk), while a moderate ratio might be optimal.
   - **Formula:** 
     $$ \text{Open Accounts Ratio} = \frac{\text{Total Open Accounts}}{\text{Total Accounts}} $$
   - **Category:** Account Monthly Payment
   - **Data Type:** Float

47. **closed_accounts_ratio**
   - **Importance:** A higher ratio can indicate a longer credit history with successfully managed debts, potentially predictive of lower risk, assuming the closures weren't due to default.
   - **Formula:** 
     $$ \text{Closed Accounts Ratio} = \frac{\text{Total Closed Accounts}}{\text{Total Accounts}} $$
   - **Category:** Account Monthly Payment
   - **Data Type:** Float

48. **performing_accounts_ratio**
   - **Importance:** A direct measure of current credit health; a higher ratio is strongly predictive of lower default risk, indicating consistency in meeting obligations..
   - **Formula:** 
     $$ \text{Performing to Total Accounts Ratio} = \frac{\text{Performing Accounts}}{\text{Total Accounts}} $$
   - **Category:** Account Monthly Payment
   - **Data Type:** Float


### Additional Suggestions
    
-   **Data Privacy:** BVN and Phone numbers should be properly hidden or anonymized 
