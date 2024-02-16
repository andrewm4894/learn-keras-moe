#%%

import os

os.environ["KERAS_BACKEND"] = "tensorflow"
# had to set this env var to avoid this issue:
# https://stackoverflow.com/questions/75583410/module-save-error-typeerror-this-dict-descriptor-does-not-support-dict
os.environ["WRAPT_DISABLE_EXTENSIONS"] = "true"

import tensorflow as tf
import pandas as pd
import keras
import keras.layers as layers
from keras.utils import FeatureSpace
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import pprint as pp
from tensorflow import Tensor
from keras.models import Model
from pandas import DataFrame
from typing import List, Dict, Tuple, Any

#%%


def dataframe_to_dataset(dataframe: DataFrame, task_cols: List[str]) -> tf.data.Dataset:
    dataframe = dataframe.copy()
    labels = {col: dataframe.pop(col) for col in task_cols}
    weights = {
        f"{col}_sample_weights": dataframe.pop(f"{col}_sample_weights")
        for col in task_cols
    }

    ds = tf.data.Dataset.from_tensor_slices(
        tensors=(dict(dataframe), labels, weights), name="df_to_ds"
    )
    ds = ds.shuffle(buffer_size=len(dataframe))

    return ds


def preprocess_dataset(
    feature_space: FeatureSpace,
    features: Dict[str, Tensor],
    labels: Dict[str, Tensor],
    weights: Dict[str, Tensor],
    task_cols: List[str],
) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], Dict[str, Tensor]]:
    adjusted_labels = {task_col: labels[task_col] for task_col in task_cols}
    sample_weights = {
        task_col: weights[f"{task_col}_sample_weights"] for task_col in task_cols
    }

    return feature_space(features), adjusted_labels, sample_weights


def expert_network(
    input_tensor: Tensor,
    dense_units: List[int] = [32],
    dropout_rate: float = 0.5,
    activation: str = "relu",
    expert_index: int = 0,  # Added an index for naming
) -> Tensor:
    x = input_tensor
    for idx, units in enumerate(dense_units):
        x = layers.Dense(
            units, activation=activation, name=f"expert_{expert_index}_dense_{idx}"
        )(x)
        x = layers.Dropout(dropout_rate, name=f"expert_{expert_index}_dropout_{idx}")(x)

    return x


def gating_network(
    input_tensor: Tensor,
    num_experts: int,
    task_name: str,
    activation: str = "softmax",
) -> Tensor:
    return layers.Dense(num_experts, activation=activation, name=f"gating_{task_name}")(
        input_tensor
    )


def combine_experts(
    experts: List[Tensor], gating_weights: Tensor, task_name: str
) -> Tensor:
    weighted_experts = [
        layers.Multiply(name=f"weighted_expert_{task_name}_{i}")(
            [gating_weights[:, i : i + 1], expert]
        )
        for i, expert in enumerate(experts)
    ]

    return layers.Add(name=f"combined_experts_{task_name}")(weighted_experts)


class ModelBuilder:
    def __init__(
        self,
        feature_space: FeatureSpace,
        num_experts: int = 2,
        experts_dense_units: List[int] = [32],
        experts_dropout_rate: float = 0.5,
        experts_activation: str = "relu",
        gating_activation: str = "softmax",
        prediction_activation: str = "sigmoid",
        task_cols: List[str] = ["task_one", "task_two"],
    ):
        self.feature_space = feature_space
        self.num_experts = num_experts
        self.experts_dense_units = experts_dense_units
        self.experts_dropout_rate = experts_dropout_rate
        self.experts_activation = experts_activation
        self.gating_activation = gating_activation
        self.prediction_activation = prediction_activation
        self.task_cols = task_cols
        self.input_features_encoded = feature_space.get_encoded_features()
        self.input_features_dict = feature_space.get_inputs()
        self.experts = [
            expert_network(
                self.input_features_encoded,
                expert_index=i,
                dense_units=self.experts_dense_units,
                dropout_rate=self.experts_dropout_rate,
                activation=self.experts_activation,
            )
            for i in range(self.num_experts)
        ]

    def build_models(self) -> Tuple[Model, Model]:
        gating_models = []
        prediction_models = []

        for task_col in self.task_cols:
            gating_task = gating_network(
                self.input_features_encoded,
                self.num_experts,
                task_col,
                self.gating_activation,
            )
            predictions_task = combine_experts(self.experts, gating_task, task_col)
            predictions_task = layers.Dense(
                1, activation=self.prediction_activation, name=task_col
            )(predictions_task)

            gating_models.append(gating_task)
            prediction_models.append(predictions_task)

        training_model = keras.Model(
            inputs=self.input_features_encoded,
            outputs=prediction_models,
            name="training_model",
        )
        inference_model = keras.Model(
            inputs=self.input_features_dict,
            outputs=prediction_models,
            name="inference_model",
        )

        return training_model, inference_model


def brier_score(y_true: Tensor, y_pred: Tensor) -> Tensor:
    return tf.reduce_mean(tf.math.square(y_true - y_pred), axis=-1)


def mae(y_true: Tensor, y_pred: Tensor) -> Tensor:
    return tf.reduce_mean(tf.abs(y_true - y_pred), axis=-1)

#%%

# params
task_cols = ["task_one", "task_two"]
val_frac = 0.2
random_seed = 1337
ds_batch_size = 32
features = {
    # Categorical features encoded as integers
    "sex": FeatureSpace.integer_categorical(num_oov_indices=0),
    "cp": FeatureSpace.integer_categorical(num_oov_indices=0),
    "fbs": FeatureSpace.integer_categorical(num_oov_indices=0),
    "restecg": FeatureSpace.integer_categorical(num_oov_indices=0),
    "exang": FeatureSpace.integer_categorical(num_oov_indices=0),
    "ca": FeatureSpace.integer_categorical(num_oov_indices=0),
    # Categorical feature encoded as string
    "thal": FeatureSpace.string_categorical(num_oov_indices=0),
    # Numerical features to discretize
    "age": FeatureSpace.float_discretized(num_bins=30),
    # Numerical features to normalize
    "trestbps": FeatureSpace.float_normalized(),
    "chol": FeatureSpace.float_normalized(),
    "thalach": FeatureSpace.float_normalized(),
    "oldpeak": FeatureSpace.float_normalized(),
    "slope": FeatureSpace.float_normalized(),
}
crosses = [
    FeatureSpace.cross(feature_names=("sex", "age"), crossing_dim=64),
    FeatureSpace.cross(
        feature_names=("thal", "ca"),
        crossing_dim=16,
    ),
]
feature_space_output_mode = "concat"
model_num_experts = 2
model_experts_dense_units = [64, 32]
model_experts_dropout_rate = 0.5
model_experts_activation = "relu"
model_gating_activation = "softmax"
model_prediction_activation = "sigmoid"
model_optimizer = "adam"
model_loss_functions = {
    "task_one": "binary_crossentropy",
    "task_two": "binary_crossentropy",
}
model_loss_weights = {"task_one": 0.05, "task_two": 0.95}
model_metrics = {
    "task_one": ["accuracy", brier_score, mae, keras.metrics.AUC()],
    "task_two": ["accuracy", brier_score, mae, keras.metrics.AUC()],
}
model_weighted_metrics = {
    "task_one": ["accuracy", brier_score],
    "task_two": ["accuracy", brier_score],
}
fit_n_epochs = 40
fit_verbose = 2

#%%

file_url = "http://storage.googleapis.com/download.tensorflow.org/data/heart.csv"
dataframe = pd.read_csv(file_url)
dataframe["task_two"] = np.where(dataframe["slope"] == 2, 1, 0)
dataframe["task_two_sample_weights"] = np.where(dataframe["exang"] == 1, 1, 0)
dataframe["task_one_sample_weights"] = 1
dataframe.rename(columns={"target": "task_one"}, inplace=True)
print(dataframe.shape)

#%%

val_dataframe = dataframe.sample(frac=val_frac, random_state=random_seed)
train_dataframe = dataframe.drop(val_dataframe.index)

print(
    "Using %d samples for training and %d for validation"
    % (len(train_dataframe), len(val_dataframe))
)

#%%

train_ds = dataframe_to_dataset(train_dataframe, task_cols)
train_ds = train_ds.batch(ds_batch_size)

val_ds = dataframe_to_dataset(val_dataframe, task_cols)
val_ds = val_ds.batch(ds_batch_size)

#%%

pp.pprint(train_ds.element_spec)

#%%

for x, y, w in train_ds.take(1):
    print("Input:\n", x)
    print("Target:\n", y)
    print("Task Weights:\n", w)

#%%

feature_space = FeatureSpace(
    features=features,
    crosses=crosses,
    output_mode=feature_space_output_mode,
)

#%%

train_ds_with_no_labels = train_ds.map(lambda x, y, z: x)
feature_space.adapt(train_ds_with_no_labels)

#%%

# get feature_space info
feature_space_info = feature_space.get_config()
# select a random feature
random_feature = np.random.choice(list(feature_space_info["features"].keys()))
print("Random feature:", random_feature)
pp.pprint(feature_space_info["features"][random_feature])

#%%

for x, y, weights in train_ds.take(1):
    preprocessed_x = feature_space(x)
    print("preprocessed_x:\n", preprocessed_x)

#%%

preprocessed_train_ds = train_ds.map(
    lambda features, labels, weights: preprocess_dataset(
        feature_space, features, labels, weights, task_cols
    ),
    num_parallel_calls=tf.data.AUTOTUNE,
)
preprocessed_train_ds = preprocessed_train_ds.prefetch(tf.data.AUTOTUNE)
pp.pprint(preprocessed_train_ds.element_spec)

preprocessed_val_ds = val_ds.map(
    lambda features, labels, weights: preprocess_dataset(
        feature_space, features, labels, weights, task_cols
    ),
    num_parallel_calls=tf.data.AUTOTUNE,
)
preprocessed_val_ds = preprocessed_val_ds.prefetch(tf.data.AUTOTUNE)
pp.pprint(preprocessed_train_ds.element_spec)

#%%

# Build the model
model_builder = ModelBuilder(
    task_cols=task_cols,
    feature_space=feature_space,
    num_experts=model_num_experts,
    experts_dense_units=model_experts_dense_units,
    experts_dropout_rate=model_experts_dropout_rate,
    experts_activation=model_experts_activation,
    gating_activation=model_gating_activation,
    prediction_activation=model_prediction_activation,
)
training_model, inference_model = model_builder.build_models()

# Compile the training model
training_model.compile(
    optimizer=model_optimizer,
    loss=model_loss_functions,
    metrics=model_metrics,
    weighted_metrics=model_weighted_metrics,
    loss_weights=model_loss_weights,
)

#%%

# print training model summary
training_model.summary()

#%%

# print model diagram
keras.utils.plot_model(
    training_model,
    show_shapes=True,
    show_layer_names=True,
    show_trainable=True,
    rankdir="TB",
)

#%%

history = training_model.fit(
    preprocessed_train_ds,
    epochs=fit_n_epochs,
    validation_data=preprocessed_val_ds,
    verbose=fit_verbose,
)

#%%

# plot loss curves
cols = [
    "loss",
    "task_one_loss",
    "task_two_loss",
    "task_one_accuracy",
    "task_two_accuracy",
]
pd.DataFrame(history.history)[cols].plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.title("Training Metrics")
plt.show()

#%%

pp.pprint(training_model.outputs)

#%%

pp.pprint(training_model.outputs[0].shape)
pp.pprint(training_model.outputs[1].shape)

#%%

pred = training_model.predict(preprocessed_val_ds)

# use zip to add scores for task one and task two into a df
df_preds_val = pd.DataFrame(
    zip(
        pred[0].flatten(),
        pred[1].flatten(),
    ),
    columns=["task_one_pred", "task_two_pred"],
)
pp.pprint(df_preds_val.head())

#%%

df_as_dict = {
    "age": [60, 60, 60, 60],
    "sex": [1, 1, 1, 1],
    "cp": [1, 1, 1, 1],
    "trestbps": [145, 145, 145, 145],
    "chol": [233, 233, 233, 233],
    "fbs": [1, 1, 1, 1],
    "restecg": [2, 2, 2, 2],
    "thalach": [150, 150, 150, 150],
    "exang": [0, 0, 0, 0],
    "oldpeak": [2.3, 2.3, 2.3, 2.3],
    "slope": [3, 3, 3, 3],
    "ca": [0, 0, 0, 0],
    "thal": ["fixed", "fixed", "fixed", "fixed"],
}

#%%

df = pd.DataFrame(df_as_dict)

ds = tf.data.Dataset.from_tensor_slices(dict(df))
ds = ds.batch(100)

#%%

predictions = inference_model.predict(ds)

#%%

sample = {
    "age": 60,
    "sex": 1,
    "cp": 1,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 2,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 3,
    "ca": 0,
    "thal": "fixed",
}

input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}
predictions = inference_model.predict(input_dict)
pp.pprint(predictions)

print(
    f"This patient had a {100 * predictions[0][0][0]:.2f}% probability of heart disease "
    f"This patient had a {100 * predictions[1][0][0]:.2f}% probability of lung disease "
)

#%%

#%%

#%%

#%%

#%%

#%%