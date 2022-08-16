# Risk-Commodities-Detection-GNN

This repository contains baseline GNN models [provided by committee] and playground for SOTA models on risk commodities detection. 

**Resources:**

- [DGL](https://github.com/dglai)
- [OpenHGNN](https://github.com/BUPT-GAMMA/OpenHGNN)
- [PyG](https://github.com/pyg-team/pytorch_geometric)

**ICDM 2022 : Risk Commodities Detection on Large-Scale E-commerce Graphs**

The competition provides a risk commodity detection dataset extracted from real-world risk scenarios at Alibaba. It requires participants to detect risky products using graph algorithms in a large-scale and heterogeneous graph with imbalanced samples.

### Dataset
**Graph Schema**

<img width="804" alt="image" src="https://user-images.githubusercontent.com/46979228/179384103-bff4b21a-b0d1-427a-8ed2-b80820e2355a.png">

#### Training Dataset

The dataset consist of edges, node features, supervised information as well as IDs of candidate items. We list the type of each column in each file as follows.

icdm2022_session1_train.zip （training data）

- icdm2022_session1_edges.csv - edge file
source_node_id int, target_node_id int, source_node_type string, target_node_type string, edge_type string

- icdm2022_session1_nodes.csv - node feature file
node_id int, node_type string, node_atts string (notice that node_atts are 256-dimensional feature vector strings with delimiter ":")

- icdm2022_session1_train_labels.csv - training labels
item_id int, label int
icdm2022_session1_test_ids.txt （candidate items）

icdm2022_session1_test_ids.txt - candidate ids
item_id int
