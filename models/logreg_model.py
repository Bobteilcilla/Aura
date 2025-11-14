def train_logreg_model():
    '''
    1. Fetch data from GCP buckets
    2. Define X and y for training and validation set
    3. Label encode y's
    4. Build pipeline
    5. Create model
    6. Grid search for optimal parameters
    7. Extract best model
    8. Save pipeline and label encoder
    9. Push pipeline and label encoder to the cloud
    '''

    df_train = pd.read_csv("gs://aura_datasets_training_validation/AURA_aug_sep_60k.csv")
    df_val = pd.read_csv("gs://aura_datasets_training_validation/AURA_validation_sep_12k.csv")
