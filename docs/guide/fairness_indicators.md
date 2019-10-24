# Using Fairness Indicators

## Overview

## Data

To run Fairness Indicators with TFMA, make sure evaluation dataset is labelled
for the features you would like to slice by.

## Model

You can use the Tensorflow Estimator class to build your model. Support for
Keras models is coming soon to TFMA. If you would like to run TFMA on a Keras
model, please see the “Model-Agnostic TFMA” section below.

After your Estimator is trained, you will need to export a saved model for
evaluation purposes. To learn more, see the
[TFMA guide](https://www.tensorflow.org/tfx/model_analysis/get_started).

## Configuring Slices

Next, define the slices you would like to evaluate on:

```python
slice_spec = [
    tfma.slicer.SingleSliceSpec(), # Overall slice
    tfma.slicer.SingleSliceSpec(columns=[slice_selection]),
]
```

## Compute Fairness Metrics

Add a Fairness Indicators callback to the `metrics_callback` list. In the
callback, you can define a list of thresholds that the model will be evaluated
at.

```python
from tensorflow_model_analysis.addons.fairness.post_export_metrics import fairness_indicators

# Build the fairness metrics. Besides the thresholds, you also can config the example_weight_key, labels_key here. For more details, please check the api.
metrics_callbacks = [
  tfma.post_export_metrics.fairness_indicators(
      thresholds=[0.1, 0.3, 0.5, 0.7, 0.9],
      )
]

eval_shared_model = tfma.default_eval_shared_model(
    eval_saved_model_path=tfma_export_dir,
    add_metrics_callbacks=metrics_callbacks)
```

Before running the config, determine whether or not you want to enable
computation of confidence intervals and k-anonymization. By default,
k-anonymization will be turned off (k = 1).

```python
compute_confidence_intervals = True
k_anonymization_count = 50
```

Run the TFMA evaluation pipeline:

```python
validate_dataset = tf.data.TFRecordDataset(filenames=[validate_tf_file])

# Run the fairness evaluation.
with beam.Pipeline() as pipeline:
  _ = (
      pipeline
      | beam.Create([v.numpy() for v in validate_dataset])
      | 'ExtractEvaluateAndWriteResults' >>
       tfma.ExtractEvaluateAndWriteResults(
                 eval_shared_model=eval_shared_model,
                 slice_spec=slice_spec,
                 compute_confidence_intervals=compute_confidence_intervals,
                 k_anonymization_count=k_anonymization_count,
                 output_path=tfma_eval_result_path)
  )
eval_result = tfma.load_eval_result(output_path=tfma_eval_result_path)
```

## Render Fairness Indicators

```python
from tensorflow_model_analysis.addons.fairness.view import widget_view

widget_view.render_fairness_indicator(eval_result)
```

More screenshots to come re: using the UI...

# Model Agnostic Evaluation
