# PyG example

### Include

+ `pyg_demo.sh`: Shell Script (include converting raw data into PyG graph, training model and generating final result)
+ `rgcn_mb_icdm.py`: PyG rgcn model
+ `format_pyg.py`: PyG data generator
+ `best_model/`: Directory to store model trained 





### How to run

+ Run shell script

```
mkdir -p /dataset/pyg_data/
cd /code/icdm_graph_competition/pyg_example/
sh pyg_demo.sh
```

+ Output

  1. PyG Graph is stored in `/dataset/pyg_data/`

     + `icdm2022_session1.pt` PyG Heterograph of session1 data

     + `icdm2022_session2.pt` PyG Heterograph of session2 data
     + `*.nodes.pyg ` Temporary cache file (Removable)

  2. Trained Model is saved in `/code/icdm_graph_competition/best_model/`

     + (Default) `1.pth` Model file
     + (Optional) change `model_id` to customize name of model file

  3. Final result of Inference is generated in `/code/icdm_graph_competition/pyg_example/`

     + `pyg_session1.json` Result of session1
     + `pyg_session2.json` Result of session2