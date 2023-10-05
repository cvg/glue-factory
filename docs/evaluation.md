# Evaluation

Glue Factory is designed for simple and tight integration between training and evaluation.
All benchmarks are designed around one principle: only evaluate on cached results. 
This enforces reproducible baselines.
Therefore, we first export model predictions for each dataset (`export`), and evaluate the cached results in a second pass (`evaluation`).

### Running an evaluation

We currently provide evaluation scripts for [MegaDepth-1500](../gluefactory/eval/megadepth1500.py), [HPatches](../gluefactory/eval/hpatches.py), and [ETH3D](../gluefactory/eval/eth3d.py).
You can run them with:

```bash
python -m gluefactory.eval.<benchmark_name> --conf "a name in gluefactory/configs/ or path" --checkpoint "and/or a checkpoint name"
```

Each evaluation run is assigned a `tag`, which can (optionally) be customized from the command line with `--tag <your_tag>`.

To overwrite an experiment, add `--overwrite`. To only overwrite the results of the evaluation loop, add `--overwrite_eval`. We perform config checks to warn the user about non-conforming configurations between runs.

The following files are written to `outputs/results/<benchmark_name>/<tag>`: 

```yaml
conf.yaml  # the config which was used
predictions.h5  # cached predictions
results.h5  # Results for each data point in eval, in the format <metric_name>: List[float]
summaries.json  # Aggregated results for the entire dataset <agg_metric_name>: float
<plots>  # some benchmarks add plots as png files here
```

Some datasets further output plots (add `--plot` to the command line).

<details>
<summary>[Configuration]</summary>

Each evaluation has 3 main configurations:

```yaml
data: 
    ...  # How to load the data. The user can overwrite this only during "export". The defaults are used in "evaluation".
model:
    ...  # model configuration: this is only required for "export".
eval: 
    ...  # configuration for the "evaluation" loop, e.g. pose estimators and ransac thresholds.
```

The default configurations can be found in the respective evaluation scripts, e.g. [MegaDepth1500](../gluefactory/eval/megadepth1500.py).

To run an evaluation with a custom config, we expect them to be in the following format ([example](../gluefactory/configs/superpoint+lightglue.yaml)):

```yaml
model:
    ... # <your model configs>
benchmarks:
    <benchmark_name1>:
        data:
            ... # <your data configs for "export">
        model:
            ... # <your benchmark-specific model configs>
        eval:
            ... # <your evaluation configs, e.g. pose estimators>
    <benchmark_name2>:
        ... # <same structure as above>
```

The configs are then merged in the following order (taking megadepth1500 as an example):

```yaml
data: 
    default < custom.benchmarks.megadepth1500.data
model:
    default < custom.model < custom.benchmarks.megadepth1500.model
eval: 
    default < custom.benchmarks.megadepth1500.eval
```

You can then use the command line to further customize this configuration.

</details>

### Robust estimators
Gluefactory offers a flexible interface to state-of-the-art [robust estimators](../gluefactory/robust_estimators/) for points and lines.
You can configure the estimator in the benchmarks with the following config structure:

```yaml
eval:
    estimator: <estimator_name>  # poselib, opencv, pycolmap, ...
    ransac_th: 0.5  # run evaluation on fixed threshold
    #or
    ransac_th: [0.5, 1.0, 1.5]  # test on multiple thresholds, autoselect best
    <extra configs for the estimator, e.g. max iters, ...>
```

For convenience, most benchmarks convert `eval.ransac_th=-1` to a default range of thresholds. 

> [!NOTE]
> Gluefactory follows the corner convention of COLMAP, i.e. the top-left corner of the top-left pixel is (0, 0).

### Visualization

We provide a powerful, interactive visualization tool for our benchmarks, based on matplotlib. 
You can run the visualization (after running the evaluations) with:
```bash
python -m gluefactory.eval.inspect <benchmark_name> <experiment_name1> <experiment_name2> ...
```

This prints the summaries of each experiment on the respective benchmark and visualizes the data as a scatter plot, where each point is the result of from a experiment on a specific data point in the dataset.

<details>

- Clicking on one of the data points opens a new frame showing the prediction on this specific data point for all experiments listed.
- You can customize the x / y axis from the navigation bar or by clicking `x` or `y`.
- Hiting `diff_only` computes the difference between `<experiment_name1>` and all other experiments.
- Hovering over a point shows lines to the results of other experiments on the same data.
- You can switch the visualization (matches, keypoints, ...) from the navigation bar or by clicking `shift+r`.
- Clicking `t` prints a summary of the eval on this data point.
- Hitting the `left` or `right` arrows circles between data points. `shift+left` opens an extra window.

When working on a remote machine (e.g. over ssh), the plots can be forwarded to the browser with the option `--backend webagg`. Note that you need to refresh the page everytime you load a new figure (e.g. when clicking on a scatter point). This part requires some more work, and we would highly appreciate any contributions!

</details>
