# DGL example

- Process session1 dataset.

```bash
mkdir -p /dataset/dgl_data/
python format_dgl.py --graph=/dataset/icdm2022_session1_edges.csv --node=/dataset/icdm2022_session1_nodes.csv --storefile=/dataset/dgl_data/icdm2022_session1
mkdir -p /dataset/dgl_data/icdm2022_session1
mv /dataset/dgl_data/icdm2022_session1.* /dataset/dgl_data/icdm2022_session1
cp /dataset/icdm2022_session1_train_labels.csv /dataset/dgl_data/icdm2022_session1/icdm2022_session1_labels.csv
cp /dataset/icdm2022_session1_test_ids.txt /dataset/dgl_data/icdm2022_session1/icdm2022_session1_test_ids.csv
```

- Process session2 dataset.

```bash
mkdir -p /dataset/dgl_data/
python format_dgl.py --graph=/dataset/icdm2022_session2/icdm2022_session2_edges.csv --node=/dataset/icdm2022_session2/icdm2022_session2_nodes.csv --storefile=/dataset/dgl_data/icdm2022_session2
mkdir -p /dataset/dgl_data/icdm2022_session2
mv /dataset/dgl_data/icdm2022_session2.* /dataset/dgl_data/icdm2022_session2
cp /dataset/icdm2022_session2/icdm2022_session2_test_ids.txt /dataset/dgl_data/icdm2022_session2/icdm2022_session2_test_ids.csv
```

- train on session1 dataset.
```bash
python entity_classify_mb_icdm.py --data-cpu --gpu 0 --n-hidden 256 --batch-size 2048 --n-bases 8 --fanout 150 --n-layers 2 --n-epoch 500 --dropout 0.3 --model-path ./session1_rgcn.pt --session session1 --lr 0.0002 --result-path result_session1_dgl.json --data-dir /dataset/dgl_data
```

- inference on session2 dataset with loaded model (trained from session1).
```bash
python entity_classify_mb_icdm.py --data-cpu --gpu 0 --n-hidden 256 --batch-size 2048 --n-bases 8 --fanout 150 --n-layers 2 --model-path ./session1_rgcn.pt --session session2 --load-model --result-path result_session2_dgl.json --data-dir /dataset/dgl_data
```
