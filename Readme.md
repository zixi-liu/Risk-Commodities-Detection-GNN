# Data Structure
##Session 1------Training Dataset

The dataset consist of edges, node features, supervised information as well as IDs of candidate items. We list the type of each column in each file as follows.

**icdm2022_session1_train.zip （training data）**
    
    1. icdm2022_session1_edges.csv ## edge file
    source_node_id int, target_node_id int, source_node_type string, target_node_type string, edge_type string
    
    2. icdm2022_session1_nodes.csv ## node feature file
    node_id int, node_type string, node_atts string (notice that node_atts are 256-dimensional feature vector strings with delimiter ":")
   
    3. icdm2022_session1_train_labels.csv ## training labels
    item_id int, label int
    

**icdm2022_session1_test_ids.txt （candidate items）**

    icdm2022_session1_test_ids.txt ## candidate ids
    item_id int
    