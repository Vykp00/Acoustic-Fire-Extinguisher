
import tensorflow as tf
import tensorflow_transform as tft

import aefire_constants

# Unpack the contents of the constants module
_DENSE_FLOAT_FEATURE_KEYS = aefire_constants.DENSE_FLOAT_FEATURE_KEYS
_BUCKET_FEATURE_KEYS = aefire_constants.BUCKET_FEATURE_KEYS
_FEATURE_BUCKET_COUNT = aefire_constants.FEATURE_BUCKET_COUNT
_RANGE_FEATURE_KEYS = aefire_constants.RANGE_FEATURE_KEYS
_VOCAB_SIZE = aefire_constants.VOCAB_SIZE
_OOV_SIZE = aefire_constants.OOV_SIZE
_VOCAB_FEATURE_KEYS = aefire_constants.VOCAB_FEATURE_KEYS
_STATUS_KEY = aefire_constants.STATUS_KEY
_transformed_name = aefire_constants.transformed_name

# Define the transformantions
def preprocessing_fn(inputs):
    """tf.transform's callback function for preprocessing inputs.
    Args:
    inputs: map from feature keys to raw not-yet-transformed features.
    Returns:
    Map from string feature key to transformed feature operations.
    """
    outputs = {}

    # Scale these features to the z-score
    for key in _DENSE_FLOAT_FEATURE_KEYS:
        outputs[_transformed_name(key)] = tft.scale_to_z_score(inputs[key])

    # Bucketize the feature
    for key in _BUCKET_FEATURE_KEYS:
        outputs[_transformed_name(key)] = tft.bucketize(inputs[key], _FEATURE_BUCKET_COUNT[key])

    # Scale these feature in range [0,1]
    for key in _RANGE_FEATURE_KEYS:
        outputs[_transformed_name(key)] = tft.scale_to_0_1(inputs[key])

    # Transform the strings into indices
    for key in _VOCAB_FEATURE_KEYS:
        outputs[_transformed_name(key)] = tft.compute_and_apply_vocabulary(
            inputs[key],
            top_k=(_VOCAB_SIZE),
            num_oov_buckets=(_OOV_SIZE))
    
    # Keep the features as is. No tft function needed
    outputs[_transformed_name(_STATUS_KEY)] = inputs[_STATUS_KEY]
    
    return outputs
