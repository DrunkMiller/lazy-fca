import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from scipy.stats import chi2


def binarize_checking_balance_feature(data, verbose=False):
    data['cb_unk'] = np.where(data['checking_balance'] == 'unknown', 1, 0)
    data['cb<0'] = np.where(data['checking_balance'] == '< 0 DM', 1, 0)
    data['cb1-200'] = np.where(data['checking_balance'] == '1 - 200 DM', 1, 0)
    data['cb>200'] = np.where(data['checking_balance'] == '> 200 DM', 1, 0)
    if verbose:
        print(data[['cb_unk', 'cb<0', 'cb1-200', 'cb>200', 'checking_balance']])
    data.drop('checking_balance', axis='columns', inplace=True)


def binarize_months_loan_duration_feature(data, verbose=False):
    data['mld<6'] = np.where(data['months_loan_duration'] <= 6, 1, 0)
    data['mld<12'] = np.where(data['months_loan_duration'] <= 12, 1, 0)
    data['mld<18'] = np.where(data['months_loan_duration'] <= 18, 1, 0)
    data['mld<24'] = np.where(data['months_loan_duration'] <= 24, 1, 0)
    data['mld>24'] = np.where(data['months_loan_duration'] > 24, 1, 0)
    if verbose:
        print(data[['mld<6', 'mld<12', 'mld<18', 'mld<24', 'mld>24', 'months_loan_duration']])
    data.drop('months_loan_duration', axis='columns', inplace=True)


def binarize_credit_history_feature(data, verbose=False):
    data['ch_r'] = np.where(data['credit_history'] == 'repaid', 1, 0)
    data['ch_c'] = np.where(data['credit_history'] == 'critical', 1, 0)
    data['ch_d'] = np.where(data['credit_history'] == 'delayed', 1, 0)
    data['ch_rb'] = np.where(data['credit_history'] == 'fully repaid this bank', 1, 0)
    data['ch_fr'] = np.where(data['credit_history'] == 'fully repaid', 1, 0)
    if verbose:
        print(data[['ch_r', 'ch_c', 'ch_d', 'ch_rb', 'ch_fr', 'credit_history']])
    data.drop('credit_history', axis='columns', inplace=True)


def binarize_amount_feature(data, verbose=False):
    data['am<1365'] = np.where(data['amount'] <= 1365, 1, 0)
    data['am<2319'] = np.where(data['amount'] <= 2319, 1, 0)
    data['am<3972'] = np.where(data['amount'] <= 3972, 1, 0)
    data['am>3973'] = np.where(data['amount'] > 3973, 1, 0)
    if verbose:
        print(data[['am<1365', 'am<2319', 'am<3972', 'am>3973', 'amount']])
    data.drop('amount', axis='columns', inplace=True)


def binarize_savings_balance_feature(data, verbose=False):
    data['sb_unk'] = np.where(data['savings_balance'] == 'unknown', 1, 0)
    data['sb<100'] = np.where(data['savings_balance'] == '< 100 DM', 1, 0)
    data['sb101-500'] = np.where(data['savings_balance'] == '101 - 500 DM', 1, 0)
    data['sb501-1000'] = np.where(data['savings_balance'] == '501 - 1000 DM', 1, 0)
    data['sb>1000'] = np.where(data['savings_balance'] == '> 1000 DM', 1, 0)
    if verbose:
        print(data[['sb_unk', 'sb<100', 'sb101-500', 'sb501-1000', 'sb>1000', 'savings_balance']])
    data.drop('savings_balance', axis='columns', inplace=True)


def binarize_employment_length_feature(data, verbose=False):
    data['el_un'] = np.where(data['employment_length'] == 'unemployed', 1, 0)
    data['el0-1'] = np.where(data['employment_length'] == '0 - 1 yrs', 1, 0)
    data['el1-4'] = np.where(data['employment_length'] == '1 - 4 yrs', 1, 0)
    data['el4-7'] = np.where(data['employment_length'] == '4 - 7 yrs', 1, 0)
    data['el>7'] = np.where(data['employment_length'] == '> 7 yrs', 1, 0)
    if verbose:
        print(data[['el_un', 'el0-1', 'el1-4', 'el4-7', 'el>7', 'employment_length']])
    data.drop('employment_length', axis='columns', inplace=True)


def binarize_installment_rate_feature(data, verbose=False):
    data['ir1'] = np.where(data['installment_rate'] == 1, 1, 0)
    data['ir2'] = np.where(data['installment_rate'] == 2, 1, 0)
    data['ir3'] = np.where(data['installment_rate'] == 3, 1, 0)
    data['ir4'] = np.where(data['installment_rate'] == 4, 1, 0)
    if verbose:
        print(data[['ir1', 'ir2', 'ir3', 'ir4', 'installment_rate']])
    data.drop('installment_rate', axis='columns', inplace=True)


def binarize_personal_status_feature(data, verbose=False):
    data['ps_male'] = np.where(data['personal_status'].str.contains(' male'), 1, 0)
    data['ps_married'] = np.where(data['personal_status'].str.contains('married'), 1, 0)
    data['ps_divorced'] = np.where(data['personal_status'].str.contains('divorced'), 1, 0)
    if verbose:
        print(data[['ps_male', 'ps_married', 'ps_divorced', 'personal_status']])
    data.drop('personal_status', axis='columns', inplace=True)


def binarize_other_debtors_feature(data, verbose=False):
    data['od_none'] = np.where(data['other_debtors'] == 'none', 1, 0)
    data['od_guar'] = np.where(data['other_debtors'] == 'guarantor', 1, 0)
    data['od_co'] = np.where(data['other_debtors'] == 'co-applicant', 1, 0)
    if verbose:
        print(data[['od_none', 'od_guar', 'od_co', 'other_debtors']])
    data.drop('other_debtors', axis='columns', inplace=True)


def binarize_residence_history_feature(data, verbose=False):
    data['rh1'] = np.where(data['residence_history'] == 1, 1, 0)
    data['rh2'] = np.where(data['residence_history'] == 2, 1, 0)
    data['rh3'] = np.where(data['residence_history'] == 3, 1, 0)
    data['rh4'] = np.where(data['residence_history'] == 4, 1, 0)
    if verbose:
        print(data[['rh1', 'rh2', 'rh3', 'rh4', 'residence_history']])
    data.drop('residence_history', axis='columns', inplace=True)


def binarize_property_feature(data, verbose=False):
    data['pr_o'] = np.where(data['property'] == 'other', 1, 0)
    data['pr_re'] = np.where(data['property'] == 'real estate', 1, 0)
    data['pr_bss'] = np.where(data['property'] == 'building society savings', 1, 0)
    data['pr_unk'] = np.where(data['property'] == 'unknown/none', 1, 0)
    if verbose:
        print(data[['pr_o', 'pr_re', 'pr_bss', 'pr_unk', 'property']])
    data.drop('property', axis='columns', inplace=True)


def binarize_age_feature(data, verbose=False):
    data['age<27'] = np.where(data['age'] <= 27, 1, 0)
    data['age<33'] = np.where(data['age'] <= 33, 1, 0)
    data['age<42'] = np.where(data['age'] <= 42, 1, 0)
    data['age>42'] = np.where(data['age'] > 42, 1, 0)
    if verbose:
        print(data[['age<27', 'age<33', 'age<42', 'age>42', 'age']])
    data.drop('age', axis='columns', inplace=True)


def binarize_installment_plan_feature(data, verbose=False):
    data['ip_none'] = np.where(data['installment_plan'] == 'none', 1, 0)
    data['ip_bank'] = np.where(data['installment_plan'] == 'bank', 1, 0)
    data['ip_stores'] = np.where(data['installment_plan'] == 'stores', 1, 0)
    if verbose:
        print(data[['ip_none', 'ip_bank', 'ip_stores', 'installment_plan']])
    data.drop('installment_plan', axis='columns', inplace=True)


def binarize_housing_feature(data, verbose=False):
    data['h_owne'] = np.where(data['housing'] == 'own', 1, 0)
    data['h_rent'] = np.where(data['housing'] == 'rent', 1, 0)
    data['h_ff'] = np.where(data['housing'] == 'for free', 1, 0)
    if verbose:
        print(data[['h_owne', 'h_rent', 'h_ff', 'housing']])
    data.drop('housing', axis='columns', inplace=True)


def binarize_existing_credits_feature(data, verbose=False):
    data['ec1'] = np.where(data['existing_credits'] == 1, 1, 0)
    data['ec2'] = np.where(data['existing_credits'] == 2, 1, 0)
    data['ec3'] = np.where(data['existing_credits'] == 3, 1, 0)
    data['ec4'] = np.where(data['existing_credits'] == 4, 1, 0)
    if verbose:
        print(data[['ec1', 'ec2', 'ec3', 'ec4', 'existing_credits']])
    data.drop('existing_credits', axis='columns', inplace=True)


def binarize_dependents_feature(data, verbose=False):
    data['dep'] = np.where(data['dependents'] == 2, 1, 0)
    if verbose:
        print(data[['dep', 'dependents']])
    data.drop('dependents', axis='columns', inplace=True)


def binarize_foreign_worker_feature(data, verbose=False):
    data['fw'] = np.where(data['foreign_worker'] == 'yes', 1, 0)
    if verbose:
        print(data[['fw', 'foreign_worker']])
    data.drop('foreign_worker', axis='columns', inplace=True)


def binarize_telephone_feature(data, verbose=False):
    data['tel'] = np.where(data['telephone'] == 'yes', 1, 0)
    if verbose:
        print(data[['tel', 'telephone']])
    data.drop('telephone', axis='columns', inplace=True)


def binarize_job_feature(data, verbose=False):
    data['j_se'] = np.where(data['job'] == 'skilled employee', 1, 0)
    data['j_ur'] = np.where(data['job'] == 'unskilled resident', 1, 0)
    data['j_ms'] = np.where(data['job'] == 'mangement self-employed', 1, 0)
    data['j_un'] = np.where(data['job'] == 'unemployed non-resident', 1, 0)
    if verbose:
        print(data[['j_se', 'j_ur', 'j_ms', 'j_un', 'job']])
    data.drop('job', axis='columns', inplace=True)


def binarize_purpose_feature(data, verbose=False):
    data['p_rt'] = np.where(data['purpose'] == 'radio/tv', 1, 0)
    data['p_cn'] = np.where(data['purpose'] == 'car (new)', 1, 0)
    data['p_f'] = np.where(data['purpose'] == 'furniture', 1, 0)
    data['p_cu'] = np.where(data['purpose'] == 'car (used)', 1, 0)
    data['p_b'] = np.where(data['purpose'] == 'business', 1, 0)
    data['p_e'] = np.where(data['purpose'] == 'education', 1, 0)
    data['p_re'] = np.where(data['purpose'] == 'repairs', 1, 0)
    data['p_d'] = np.where(data['purpose'] == 'domestic appliances', 1, 0)
    data['p_o'] = np.where(data['purpose'] == 'others', 1, 0)
    data['p_r'] = np.where(data['purpose'] == 'retraining', 1, 0)
    if verbose:
        print(data[['p_rt', 'p_cn', 'p_f', 'p_cu', 'p_b', 'p_e', 'p_re', 'p_d', 'p_o', 'p_r', 'purpose']])
    data.drop('purpose', axis='columns', inplace=True)


def binarize_target_feature(data, verbose=False):
    data['credit_approval'] = np.where(data['default'] == 2, 1, 0)
    if verbose:
        print(data[['credit_approval', 'default']])
    data.drop('default', axis='columns', inplace=True)


def dependency_check(crosstable, prob=0.95):
    stat, p, dof, expected = chi2_contingency(crosstable)
    critical = chi2.ppf(prob, dof)
    print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))
    if abs(stat) >= critical:
        print('Dependent (reject H0)')
    else:
        print('Independent (fail to reject H0)')
    # interpret p-value
    alpha = 1.0 - prob
    print('significance=%.3f, p=%.3f' % (alpha, p))
    if p <= alpha:
        print('Dependent (reject H0)')
    else:
        print('Independent (fail to reject H0)')


def check_dependency_by_all_features(data, target_feature_name, prob):
    for feature_name in list(data):
        if target_feature_name == feature_name:
            continue
        start_str = '------- ' + feature_name + ' -------'
        end_str = ''
        for i in range(len(start_str)):
            end_str += '-'
        print(start_str)
        dependency_check(pd.crosstab(data[feature_name], data[target_feature_name]))
        print(end_str)


def col_statistics(data, name):
    data[name].value_counts().plot.bar()
    print(data[name].value_counts())
    print(data[name].describe())
    plt.show()


def all_col_statistics(data):
    for col_name in list(data):
        start_str = '------- ' + col_name + ' -------'
        end_str = ''
        for i in range(len(start_str)):
            end_str += '-'
        print(start_str)
        col_statistics(data, col_name)
        print(end_str)


def preparing_data(features, data):
    data_tmp = data.copy()
    features_to_delete = list(features_binarize_dict.keys())
    for feature in features:
        features_to_delete.remove(feature)
        features_binarize_dict[feature](data_tmp, False)
    for feature in features_to_delete:
        data_tmp.drop(feature, axis='columns', inplace=True)
    return data_tmp


features_binarize_dict = {
    'months_loan_duration': binarize_months_loan_duration_feature,
    'checking_balance': binarize_checking_balance_feature,
    'credit_history': binarize_credit_history_feature,
    'purpose': binarize_purpose_feature,
    'amount': binarize_amount_feature,
    'savings_balance': binarize_savings_balance_feature,
    'employment_length': binarize_employment_length_feature,
    'installment_rate': binarize_installment_rate_feature,
    'personal_status': binarize_personal_status_feature,
    'other_debtors': binarize_other_debtors_feature,
    'residence_history': binarize_residence_history_feature,
    'property': binarize_property_feature,
    'age': binarize_age_feature,
    'installment_plan': binarize_installment_plan_feature,
    'housing': binarize_housing_feature,
    'existing_credits': binarize_existing_credits_feature,
    'dependents': binarize_dependents_feature,
    'telephone': binarize_telephone_feature,
    'foreign_worker': binarize_foreign_worker_feature,
    'job': binarize_job_feature,
    'default': binarize_target_feature,
}


finall_features = {
    'months_loan_duration',
    'checking_balance',
    'credit_history',
    'purpose',
    #'amount',
    #'savings_balance',
    'employment_length',
    'installment_rate',
    'personal_status',
    'other_debtors',
    'residence_history',
    'property',
    #'age',
    #'installment_plan',
    #'housing',
    'existing_credits',
    #'dependents',
    'telephone',
    #'foreign_worker',
    #'job',
    'default',
}


if __name__ == "__main__":
    pd.options.mode.chained_assignment = None
    # pd.set_option("display.max_rows", None, "display.max_columns", None)

    data = pd.read_csv('D:\credit.csv', delimiter=';')
    bin_data = preparing_data(finall_features, data)
    bin_data.to_csv('binarized_credit.csv', index=False, sep=';')

    #check_dependency_by_all_features(data, 'default', 0.98)

    # col_statistics(data, 'installment_plan')
    #col_statistics(data, 'dependents')
