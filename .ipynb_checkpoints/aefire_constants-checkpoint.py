
# Features with string data types that will be converted to indices
CATEGORICAL_FEATURE_KEYS = ['FUEL']

# Numerical features that are marked as continuous
NUMERIC_FEATURE_KEYS = ['DESIBEL','FREQUENCY','SIZE']

# Feature that can be grouped into buckets
BUCKET_FEATURE_KEYS = ['AIRFLOW', 'DISTANCE']

# Number of buckets used by tf.transform for encoding each bucket feature.
FEATURE_BUCKET_COUNT = {'AIRFLOW': 5, 'DISTANCE': 5}

# Feature that the model will predict
LABEL_KEY = 'STATUS'

# Utility function for renaming the feature
def transformed_name(key):
    return key + '_xf'
