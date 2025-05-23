# cond-dp
Private Learning with Public Feature Conditioning

#### Major Package Requirements

- PyTorch
- Opacus (for differential privacy)
- absl-py (for argument parsing)

#### File Descriptions
- Main script: ``private_lin_reg.py``— Entry point for running all the experiments.
- Data loading: ``data_utils.py``
- Privacy utilities: ``privacy_utils.py``
- Example launcher: ``run_exp_wine.py``— Demonstrates how to run an experiment on the Wine dataset 
with 128 epochs and the privacy loss $\epsilon=1$.

To run experiments, use:
```angular2html
$python private_lin_reg.py [arguments]
```

#### Dataset Preparation
Each dataset folder should include the following NumPy files:
- ``X_train.npy``: training features
- ``y_train.npy``: training labels
- ``X_test.npy``: test features
- ``y_test.npy``: test labels

If using Cond-DP with a precomputed conditioning matrix ``C``, save ``C``
in a numpy file, such as ``C.npy``.

#### Command-Line Arguments for ``private_lin_reg.py``:

All command-line arguments are documented within the script, 
and most are self-explanatory. Key flags include:
- ``--train_batch_size``/``--test_batch_size``: Set to ``None`` to use full-batch gradient descent.
- ``--precond_matrix_path``: Path to a conditioning matrix file. 
Default to ``None``, which corresponds to standard DPGD/DPSGD (no preconditioning). Set this to use a specific matrix for Cond-DP. 
- ``--mlp_layers``: Specifies the hidden layer sizes in an MLP-based prediction layer.
Default to ``None``, which corresponds to private linear regression without MLP layers.  
- ``--data_dir``: Path to the dataset folder that contains the four files mentioned above.