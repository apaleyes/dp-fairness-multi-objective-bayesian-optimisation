import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

# AIF360 imports
from aif360.datasets import AdultDataset, MEPSDataset21

def load_adult():
    attributte_names =  \
        [
            "age", "workclass", "fnlwgt", "education", "education-num",  
            "marital status", "occupation", "relationship", "race", "sex", 
            "capital gain", "capital loss", "hours per week", "country", "income-per-year"
        ]

    # Dataset downloaded from https://archive.ics.uci.edu/ml/datasets/adult
    adult_data_train = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", header = None, names = attributte_names, skipinitialspace=True, na_values = ['?'])
    adult_data_test = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test", header = 0, names = attributte_names, skipinitialspace=True, na_values = ['?'])

    adult_data_train = adult_data_train.dropna()
    adult_data_test = adult_data_test.dropna()

    return (adult_data_train, adult_data_test)

from sklearn.model_selection import train_test_split

# Preprocessing based on the https://arxiv.org/pdf/2102.05975.pdf paper
def preprocess_adult_paper_based():
    (adult_data_train, adult_data_test) = load_adult()

    # Map the income data and sex attribute into numerical indicators
    # Note: Use 1 for male and 0 for female because the default behaviour of AIF360 is to consider the highest value as privileged
    adult_data_train['income-per-year'] = adult_data_train['income-per-year'].map({'<=50K': 0, '>50K': 1}).astype(int)
    adult_data_train['sex'] = adult_data_train['sex'].map({'Male': 1, 'Female': 0}).astype(int)

    adult_data_test['income-per-year'] = adult_data_test['income-per-year'].map({'<=50K.': 0, '>50K.': 1}).astype(int)
    adult_data_test['sex'] = adult_data_test['sex'].map({'Male': 1, 'Female': 0}).astype(int)

    # One hot encoding for all categorical values
    one_hot_encoder = OneHotEncoder()
    categorical_columns = ['marital status', 'occupation', 'relationship', 'race']

    train_categorical_columns_encoder_df = pd.DataFrame(one_hot_encoder.fit_transform(adult_data_train[categorical_columns]).toarray())
    train_categorical_columns_encoder_df.columns = one_hot_encoder.get_feature_names()
    train_categorical_columns_encoder_df.index = adult_data_train[categorical_columns].index

    test_categorical_columns_encoder_df = pd.DataFrame(one_hot_encoder.transform(adult_data_test[categorical_columns]).toarray())
    test_categorical_columns_encoder_df.columns = one_hot_encoder.get_feature_names()
    test_categorical_columns_encoder_df.index = adult_data_test[categorical_columns].index

    adult_data_train = pd.concat([train_categorical_columns_encoder_df, adult_data_train.drop(columns = categorical_columns)], axis = 1)
    adult_data_test = pd.concat([test_categorical_columns_encoder_df, adult_data_test.drop(columns = categorical_columns)], axis = 1)

    # Normalize the continuous variables
    min_max_scaler = MinMaxScaler()
    continuous_columns = ['age', 'capital gain', 'education-num', 'capital loss', 'hours per week']

    adult_data_train = adult_data_train.drop(['education', 'fnlwgt', 'country', 'workclass'], axis = 1)
    adult_data_test = adult_data_test.drop(['education', 'fnlwgt', 'country', 'workclass'], axis = 1)

    adult_data_train[continuous_columns] = min_max_scaler.fit_transform(adult_data_train[continuous_columns])
    adult_data_test[continuous_columns] = min_max_scaler.transform(adult_data_test[continuous_columns])

    training_data, validation_data = train_test_split(adult_data_train, test_size = 0.2, shuffle = False)

    return (df_to_binary_label_dataset(training_data, protected_attributes_list = ['sex'], target_attribute = 'income-per-year'),
            df_to_binary_label_dataset(validation_data, protected_attributes_list = ['sex'], target_attribute = 'income-per-year'),
            df_to_binary_label_dataset(adult_data_test, protected_attributes_list = ['sex'], target_attribute = 'income-per-year'))

# Preprocessing similar to AIF360 
def preprocess_adult_aif360_based():
    (adult_data_train, adult_data_test) = load_adult()

    def preprocess_df(df):
        df['age (decade)'] = df['age'].apply(lambda x: x//10*10)

        def group_edu(x):
            if x <= 5:
                return 5
            elif x >= 13:
                return 13
            else:
                return x

        def age_cut(x):
            if x >= 70:
                return 70
            else:
                return x

        def group_race(x):
            if x == "White":
                return 1.0
            else:
                return 0.0

        # Limit education range
        df['education years'] = df['education-num'].apply(lambda x : group_edu(x))
        df['education years'] = df['education years'].astype('category')

        # Limit age range
        df['age (decade)'] = df['age (decade)'].apply(lambda x : age_cut(x))

        # Transform all that is non-white into 'minority'
        df['race'] = df['race'].apply(lambda x: group_race(x))

        # Replace income with binary variable
        df['income-per-year'] = df['income-per-year'].replace(to_replace='>50K.', value='>50K', regex=True)
        df['income-per-year'] = df['income-per-year'].replace(to_replace='<=50K.', value='<=50K', regex=True)
        df['income-per-year'] = df['income-per-year'].map({'<=50K': 0.0, '>50K': 1.0})

        # Add binary sex variable
        df['sex'] = df['sex'].map({'Female': 0.0, 'Male': 1.0})

        features = ['age (decade)','education years','race','sex','income-per-year']
        return df[features]

    preprocessed_train = preprocess_df(adult_data_train)
    preprocessed_test = preprocess_df(adult_data_test)

    scaler = MinMaxScaler(copy=False)

    train_binary_label_dataset = df_to_binary_label_dataset(preprocessed_train, protected_attributes_list = ['sex'], target_attribute = 'income-per-year')
    test_binary_label_dataset = df_to_binary_label_dataset(preprocessed_test, protected_attributes_list = ['sex'], target_attribute = 'income-per-year')

    train_binary_label_dataset.features = scaler.fit_transform(train_binary_label_dataset.features)
    test_binary_label_dataset.features = scaler.fit_transform(test_binary_label_dataset.features)

    ## TODO: Print train_binary_label_dataset.features and targets?
    return (train_binary_label_dataset, test_binary_label_dataset)

from utils import write_dataset_to_file, write_to_log_file

def load_ADULT_from_AIF(should_scale = True, validation = False, log_file = None, use_all_features = True):
    ## TODO: Check this, because I am now using all the features
    all_features_in_adult = ['age', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
    prev_features_used = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

    ###################################################################################
    ### For some reason using one hot encoding signficantly affects the performance ###
    ###################################################################################
    if use_all_features:
        ad = AdultDataset(protected_attribute_names=['sex'],
        privileged_classes = [['Male']],
        categorical_features=['workclass', 'education',
                            'marital-status', 'occupation', 'relationship',
                            'native-country', 'race'],
        features_to_keep = all_features_in_adult)
    else:
        ad = AdultDataset(protected_attribute_names=['sex'],
        privileged_classes=[['Male']], categorical_features=[],
        features_to_keep=['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week'])

    # write_dataset_to_file(ad.convert_to_dataframe()[0], '/home/bf319/new-Pareto/ParetoFronts/adult_all_features.csv')

    # ad = AdultDataset(protected_attribute_names=['sex'],
    # privileged_classes=[['Male']], categorical_features=[],
    # features_to_keep=['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week'])

    if validation:
        train_binary_label_dataset, valid_test_binary_label_dataset = ad.split([0.533], shuffle=True)
        valid_binary_label_dataset, test_binary_label_dataset = valid_test_binary_label_dataset.split([0.7136], shuffle = True)

        if should_scale:
            scaler = MinMaxScaler(copy=False)

            train_binary_label_dataset.features = scaler.fit_transform(train_binary_label_dataset.features)
            valid_binary_label_dataset.features = scaler.transform(valid_binary_label_dataset.features)
            test_binary_label_dataset.features = scaler.transform(test_binary_label_dataset.features)

        return (train_binary_label_dataset, valid_binary_label_dataset, test_binary_label_dataset)
    else:
        train_binary_label_dataset, test_binary_label_dataset = ad.split([0.8], shuffle=True)
        if should_scale:
            scaler = MinMaxScaler(copy=False)

            train_binary_label_dataset.features = scaler.fit_transform(train_binary_label_dataset.features)
            test_binary_label_dataset.features = scaler.transform(test_binary_label_dataset.features)
        
        return (train_binary_label_dataset, test_binary_label_dataset)

def get_full_adult_dataset():
    all_features_in_adult = ['age', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']

    ad = AdultDataset(protected_attribute_names=['sex'],
        privileged_classes = [['Male']],
        categorical_features=['workclass', 'education',
                            'marital-status', 'occupation', 'relationship',
                            'native-country', 'race'],
        features_to_keep = all_features_in_adult)

    scaler = MinMaxScaler(copy=False)

    ad.features = scaler.fit_transform(ad.features)
    return ad

from aif360.datasets import BinaryLabelDataset

def df_to_binary_label_dataset(df, protected_attributes_list, target_attribute):
    return BinaryLabelDataset(
        favorable_label = 1.0,
        unfavorable_label = 0.0,
        ## Below args for StructuredDataset
        df = df,
        # feature_names = list(df.columns).remove(target_attribute),
        label_names = [target_attribute],
        protected_attribute_names = protected_attributes_list,
    )

from aif360.datasets import CompasDataset, GermanDataset

def load_compas():
    cd = CompasDataset()

    write_dataset_to_file(cd.convert_to_dataframe()[0], '/home/bf319/new_experiments_Pareto/compass_full.csv')

    (cd_train, cd_test) = cd.split([0.8], shuffle = False)

    return (cd_train, cd_test)

def load_meps():
    meps = MEPSDataset21(categorical_features = [])

    (meps_train, meps_test) = meps.split([0.8], shuffle = False)

    scaler = MinMaxScaler(copy=False)

    meps_train.features = scaler.fit_transform(meps_train.features)
    meps_test.features = scaler.transform(meps_test.features)

    write_dataset_to_file(meps.convert_to_dataframe()[0], '/home/bf319/new_experiments_Pareto/meps.csv')

    return (meps_train, meps_test)