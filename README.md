# Local Reference Frame Features for Graphs

This repository contains an implementation of Local Reference Frame (LRF) and LRF-based features SHOT and ROPS.
The code is entirely implemented in pytorch, with helper functions from pytorch3d library.
Training is done using pytorch lightning library.

Notably, this implementation supports the following features:

- Direct integration with pytorch and pytorch-geometric.
- Completely vectorised for efficiency.
- Supports batching (pytorch-geometric style).
- Supports GPU and multi-GPU training.

## Usage

`main.py` exposes a command line interface, with the following signature:

```
python main.py <args>
```

For example, the following runs the GCN model with ROPS feature on GPU 0.
The training will be logged by tensorboard at `TRAIN_PATH`.

```
python main.py --model gcn --lrf rops --gpu_id 0
```

Arguments (see `main.py` for defaults):

- `model : gcn/gat` The model type
- `lrf : none/shot/rops` The LRF feature
- `n_neighbors : int` Number of neighbors to use for LRF feature calculation
- `lr : float` Learning rate
- `batch_size : int` Batch size
- `label : int` QM9's label id
- `max_epochs : int` Number of maximum epochs to train for. Note that training is stopped when validation loss converges.
- `gpu_id : int` By default, pytorch lightning will try its best to make use of the resource. If there are more than 1 GPU, it will run distributed training. Set this to disable this default behavior and run only on the selected GPU.
- `profiler : none/simple` Pytorch lightning's profiler to use.

## Project Outline

- `lrf.py` implements the Local Reference Frame Reference. It returns the LRF associated with each point.
- `shot.py` and `rops.py` implements the SHOT and ROPS descriptor.
- `model.py` contains the GCN and GATv2 implementations and `training.py` implements the wrapper to train and evaluate the corresponding models.
- `data.py` and `utils.py` contains additional utilities to load and prepare the datasets.
