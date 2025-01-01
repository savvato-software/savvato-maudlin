import pandas as pd
from keras.saving import register_keras_serializable

@register_keras_serializable(package="CustomPackage")
def apply(data, columns):

    data.drop(columns=columns, inplace=True)

    return data

