from openhgnn import Experiment
from openhgnn.dataset import AsNodeClassificationDataset, ICDM2022Dataset
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

if __name__ == '__main__':
    myNCDataset = AsNodeClassificationDataset(ICDM2022Dataset(session='session1', force_reload=True, load_labels=True),
                                              target_ntype='item', force_reload=True)

    experiment = Experiment(specified_trainer='icdm_trainer', model='RGCN', dataset=myNCDataset,
                            task='node_classification', gpu=0, mini_batch_flag=True, max_epoch=0,
                            load_from_pretrained=True, data_cpu=True, batch_size=512)
    experiment.run()
