
# Features to be scaled to the z-score
DENSE_FLOAT_FEATURE_KEYS = ['AIRFLOW']

# Feature that can be grouped into buckets
BUCKET_FEATURE_KEYS = ['DISTANCE']

# Number of buckets used by tf.transform for encoding each feature.
FEATURE_BUCKET_COUNT = {'DISTANCE': 3}

# Feature to scale from 0 to 1
RANGE_FEATURE_KEYS = ['DESIBEL', 'FREQUENCY', 'SIZE']

# Number of vocabulary terms used for encoding VOCAB_FEATURES by tf.transform
VOCAB_SIZE = 10

# Count of out-of-vocab buckets in which unrecognized VOCAB_FEATURES are hashed.
OOV_SIZE = 5

# Features with string data types that will be converted to indices
VOCAB_FEATURE_KEYS = ['FUEL']

# Feature that the model will predict
STATUS_KEY = 'STATUS'

# Utility function for renaming the feature
def transformed_name(key):
    return key + '_xf'
