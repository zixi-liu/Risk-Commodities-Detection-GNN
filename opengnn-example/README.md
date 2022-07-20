# OpenHGNN example

- Dataset process
Since openhgnn is based on DGL, so the dataset processing of openhgnn is identical to DGL. Refer to [DGL EXAMPLE](../dgl_example/README.md)

- Install openhgnn:

```bash
git clone -b icdm https://github.com/BUPT-GAMMA/OpenHGNN
cd OpenHGNN
pip install .
```

- train on session1 dataset.
```bash
python session1.py
```

- inference on session2 dataset with loaded model (trained from session1).
```bash
python session2.py
```

- Parameter Descriptions:
  - Parameters in ICDM2022Dataset:
      - session: The dataset name in the competition, options: "session1", "session2".
      - force_reload: Whether to reload the dataset. Default: False
      - load_labels: Whether to load labels to the graph. For "session2", there is no labels. Default: True
  - Parameters in Experiment:
      - model: The model you use in OpenHGNN, default: "RGCN".
      - dataset: The dataset of "AsNodeClassificationDataset".
      - task: "node_classification" do not change.
      - gpu: The ID of GPU you use. If you want to use cpu, please set to -1.
      - mini_batch_flag: We suggest to set it "True", as the dataset is large.
      - max_epoch: The epoch you use when training.
      - load_from_pretrained: If you already have a .pt file, you can put it under directory ./openhgnn/output/{model} .
      - data_cpu: Whether data is loaded to the CPU to save GPU resources.
      - batch_size: Used when "mini_batch_flag" is "True".
      - direct_inference: Skip the training process and make inferences directly. Please set "load_from_pretrained" to "True". 

- Change "session" to change the dataset.
- If your "session" is "session2", please make sure that "load_labels" is "False", "direct_inference" is "True".