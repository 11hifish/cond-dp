import numpy as np
from absl import app
from absl import flags

from opacus import PrivacyEngine
from opacus.accountants.rdp import RDPAccountant
from opacus.validators import ModuleValidator
import torch
import torch.nn as nn
import torch.optim as optim

from privacy_utils import get_noise_multiplier
from data_utils import load_dataset, load_precond_matrix

_DEVICE = 'cpu' if not torch.cuda.is_available() else 'cuda'

_DATA_DIR = flags.DEFINE_string(
    'data_dir',
    None,
    'Dataset folder.',
    required=True,
)
_PRECOND_MATRIX_PATH = flags.DEFINE_string(
    'precond_matrix_path',
    None,
    'Path to the pre-conditioning matrix.',
)
_TRAIN_BATCH_SIZE = flags.DEFINE_integer(
    'train_batch_size',
    # 1000,
    None,
    'Training batch size.',
)
_TEST_BATCH_SIZE = flags.DEFINE_integer(
    'test_batch_size',
    None,
    'Test batch size.',
)
_NUM_EPOCHS = flags.DEFINE_integer(
    'num_epochs',
    32,
    'Number of epochs.',
)
_LEARNING_RATE = flags.DEFINE_float(
    'learning_rate',
    0.001,
    'Learning rate.',
)
_EPSILON = flags.DEFINE_float(
    'epsilon',
    None,
    'Epsilon.',
)
_DELTA = flags.DEFINE_float(
    'delta',
    1e-6,
    'Delta.',
)
_CLIP_NORM = flags.DEFINE_float(
    'clip_norm',
    None,
    'Clip norm.',
)
_OPT_NAME = flags.DEFINE_string(
    'opt_name',
    'sgd',
    'Optimizer name. Currently support "sgd" and "adam", private and non-private. '
)
_EXP_NO = flags.DEFINE_integer(
    'exp_no',
    None,
    'Experiment number. Only used for saving results.',
)
_RES_DIR = flags.DEFINE_string(
    'res_dir',
    None,
    'Directory to save results.',
)
_MLP_LAYERS = flags.DEFINE_list(
    'mlp_layers',
    [],
    'List of comma separated layer widths of intermediate layers. Each width '
    'represents one layer. For example 1024,512 for two hidden layers with '
    '1024 and 512 neurons. The last layer has always width 1 and is added '
    'automatically.',
)
_INIT_STD = flags.DEFINE_float(
    'init_std',
    None,
    'Standard deviation of the initial weights. If provided, the linear layer'
    'weights are initialized with a normal distribution with this std.'
    'Otherwise, the weights are initialized using the default uniform init.',
)
_OUTPUT_DIM = flags.DEFINE_integer(
    'output_dim',
    1,
    'Output dimension. If 1, used for regression. '
    'If > 1, used for classification and a softmax layer is added.'
    'Only used when MLP layers are provided.',
)
_LOSS_NAME = flags.DEFINE_string(
    'loss_name',
    'mse',
    'Loss function name.',
)
_SCALING = flags.DEFINE_string(
    'scaling',
    None,
    'How to scale the data / precond matrix. "adaptive" learns a scaler. "fixed" uses average norm of X_train before '
    'and after being conditioned. Default is None, which means no rescaling at all. '
)
_MODEL_SAVE_PATH = flags.DEFINE_string(
    'model_save_path',
    None,
    'Path to save the last iterate model. Default is None, no model is saved.'
)
_MODEL_LOAD_PATH = flags.DEFINE_string(
    'model_load_path',
    None,
    'Path to load an existing model. Default is None, no model is loaded.'
)

class ScaleLayer(nn.Module):
    def __init__(self, ratio=1., num_features=1):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(num_features) * ratio)  # Can be a scalar or per-feature

    def forward(self, x):
        return self.scale * x


def main(argv) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  # get data
  data_dir = _DATA_DIR.value
  X_train, y_train, X_test, y_test = load_dataset(data_dir)
  X_train, y_train = torch.from_numpy(X_train).to(
      torch.float32
  ), torch.from_numpy(y_train).to(torch.float32)
  X_test, y_test = torch.from_numpy(X_test).to(torch.float32), torch.from_numpy(
      y_test
  ).to(torch.float32)

  # load pre-conditioning matrix
  precond_matrix_path = _PRECOND_MATRIX_PATH.value
  scaling_layer = None

  if precond_matrix_path:
    precond_matrix = load_precond_matrix(precond_matrix_path)
    # compute the starting rescaling factor of the features when doing pre-conditioning
    avg_norm_sq_X_train = np.mean(np.linalg.norm(X_train, axis=1) ** 2)
    X_cond = X_train @ precond_matrix
    avg_norm_sq_X_cond = np.mean(np.linalg.norm(X_cond, axis=1) ** 2)
    ratio = np.sqrt(avg_norm_sq_X_train / avg_norm_sq_X_cond)
    if _SCALING.value == 'fixed':
        print(f'rescaling ratio: {ratio}')
        X_train = X_train * ratio
        X_test = X_test * ratio
    elif _SCALING.value == 'learned':
        print(f'starting rescaling ratio: {ratio}')
        scaling_layer = ScaleLayer(ratio=ratio, num_features=1)
  else:
    precond_matrix = None

  # get data loader
  if _TRAIN_BATCH_SIZE.value is None:
    train_batch_size = X_train.shape[0]
  else:
    train_batch_size = _TRAIN_BATCH_SIZE.value
  if _TEST_BATCH_SIZE.value is None:
    test_batch_size = X_test.shape[0]
  else:
    test_batch_size = _TEST_BATCH_SIZE.value

  train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
  test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
  train_data_loader = torch.utils.data.DataLoader(
      train_dataset, batch_size=train_batch_size
  )
  test_data_loader = torch.utils.data.DataLoader(
      test_dataset, batch_size=test_batch_size
  )

  def get_lin_reg_model(input_dim, precond_matrix=None, mlp_layers=None):
    if mlp_layers:
      output_dim = mlp_layers[0]
    else:
      output_dim = 1
    print(f'Output dim of the linear layer is: {output_dim}')
    if precond_matrix is not None:  # use pre-conditioning
      assert precond_matrix.shape[0] == precond_matrix.shape[1]
      assert precond_matrix.shape[0] == input_dim
      precond_layer = nn.Linear(
          in_features=input_dim, out_features=input_dim, bias=False
      )
      precond_matrix_tensor = torch.from_numpy(precond_matrix).to(torch.float32)
      precond_layer.weight.data = precond_matrix_tensor
      # freeze precond layer
      for param in precond_layer.parameters():
        param.requires_grad = False
      linear_layer = nn.Linear(
          in_features=input_dim, out_features=output_dim, bias=True
      )
      # change initialization
      if _INIT_STD.value is not None:
        torch.nn.init.normal_(linear_layer.weight, mean=0, std=_INIT_STD.value)
      if scaling_layer:
        model = torch.nn.Sequential(precond_layer, scaling_layer, linear_layer)
      else:
        model = torch.nn.Sequential(precond_layer, linear_layer)
    else:  # vanilla dp-sgd
      linear_layer = nn.Linear(
          in_features=input_dim, out_features=output_dim, bias=True
      )
      # change initialization
      if _INIT_STD.value is not None:
        torch.nn.init.normal_(linear_layer.weight, mean=0, std=_INIT_STD.value)
      model = torch.nn.Sequential(linear_layer)
    if mlp_layers:
      in_features = output_dim
      for d in mlp_layers:
        hidden_layer = nn.Linear(
            in_features=in_features, out_features=d, bias=True
        )
        model.append(hidden_layer)
        model.append(nn.ReLU())
        in_features = d
      # define the last output layer
      model.append(
          nn.Linear(
              in_features=in_features, out_features=_OUTPUT_DIM.value, bias=True
          )
      )
    return model.to(_DEVICE)

  def get_loss_fn(loss_name='mse'):
    if loss_name == 'mse':
      criterion = nn.MSELoss()
    elif loss_name == 'ce':  # classification loss
      criterion = nn.CrossEntropyLoss()
    else:
      raise ValueError(f'Loss {loss_name} not supported!')
    return criterion

  def get_optimizer(
      model,
      optimizer_name='adam',
      learning_rate=0.001,
      epsilon=None,
      noise_multiplier=None,
      data_loader=None,
      grad_norm=None,
  ):
    # get optimizer
    if optimizer_name == 'adam':
      print('Using adam!')
      optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'sgd':
      optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    else:
      raise ValueError(f'Optimizer {optimizer_name} not supported!')
    if not epsilon:  # public setting
      print('Created optimizer in non-private settings!')
      return model, optimizer, data_loader
    elif epsilon and (
        data_loader is None or noise_multiplier is None or grad_norm is None
    ):
      raise ValueError(
          'Data loader, noise multiplier and clip norm must be present to use'
          ' private optimizer!'
      )
    # make optimizer private optimizer
    privacy_engine = PrivacyEngine()
    model, optimizer, data_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        max_grad_norm=grad_norm,
        noise_multiplier=noise_multiplier,
    )
    return model, optimizer, data_loader

  def train(
      model,
      train_data_loader,
      test_data_loader,
      optimizer,
      criterion,
      num_epochs=10,
  ):
    eval_loss_epochs = []
    eval_accuracy_epochs = []
    train_loss_epochs = [0]
    # evaluate model before training begins
    model.eval()
    eval_loss_this_epoch = 0
    n_correct_predictions = 0
    for batch_idx, (X_batch, y_batch) in enumerate(test_data_loader):
      if _OUTPUT_DIM.value > 1:  # classification
        y_batch = y_batch.type(torch.LongTensor)
      X_batch, y_batch = X_batch.to(_DEVICE), y_batch.to(_DEVICE)
      y_pred = model(X_batch)
      if _OUTPUT_DIM.value == 1:  # regression
        y_pred = y_pred.ravel()
      loss = criterion(y_pred, y_batch)
      eval_loss_this_epoch += loss.item() * len(X_batch)
      print(f'Epoch 0, batch {batch_idx}, test loss: {loss:.4f}')
      if _OUTPUT_DIM.value > 1:  # classification
        pred_class = torch.argmax(y_pred, dim=1)
        correct_predictions = (pred_class == y_batch).sum().item()
        n_correct_predictions += correct_predictions
        acc_batch = correct_predictions / len(y_batch)
        print(f'Epoch 0, batch {batch_idx}, accuracy: {acc_batch:.4f}')
    eval_loss_this_epoch = eval_loss_this_epoch / len(test_dataset)
    eval_loss_epochs.append(eval_loss_this_epoch)
    if _OUTPUT_DIM.value > 1:  # classification
      eval_accuracy_this_epoch = n_correct_predictions / len(y_test)
      eval_accuracy_epochs.append(eval_accuracy_this_epoch)

    for epoch_idx in range(num_epochs):
      model.train()
      train_loss_this_epoch = 0
      for batch_idx, (X_batch, y_batch) in enumerate(train_data_loader):
        optimizer.zero_grad()
        if _OUTPUT_DIM.value > 1:  # classification
          y_batch = y_batch.type(torch.LongTensor)
        X_batch, y_batch = X_batch.to(_DEVICE), y_batch.to(_DEVICE)
        y_pred = model(X_batch)
        if _OUTPUT_DIM.value == 1:  # regression
          y_pred = y_pred.ravel()
        loss = criterion(y_pred, y_batch)
        train_loss_this_epoch += loss.detach().item() * len(X_batch)

        loss.backward()
        optimizer.step()
      train_loss_this_epoch = train_loss_this_epoch / len(train_dataset)
      train_loss_epochs.append(train_loss_this_epoch)

      # evaluate on test data
      model.eval()
      eval_loss_this_epoch = 0
      n_correct_predictions = 0
      for batch_idx, (X_batch, y_batch) in enumerate(test_data_loader):
        if _OUTPUT_DIM.value > 1:  # classification
          y_batch = y_batch.type(torch.LongTensor)
        X_batch, y_batch = X_batch.to(_DEVICE), y_batch.to(_DEVICE)
        y_pred = model(X_batch)
        if _OUTPUT_DIM.value == 1:  # regression
          y_pred = y_pred.ravel()
        loss = criterion(y_pred, y_batch)
        eval_loss_this_epoch += loss.item() * len(X_batch)
        print(
            f'Epoch {epoch_idx + 1}, batch {batch_idx}, test loss: {loss:.4f}'
        )
        if _OUTPUT_DIM.value > 1:  # classification
          pred_class = torch.argmax(y_pred, dim=1)
          correct_predictions = (pred_class == y_batch).sum().item()
          n_correct_predictions += correct_predictions
          acc_batch = correct_predictions / len(y_batch)
          print(f'Epoch {epoch_idx + 1}, batch {batch_idx}, accuracy: {acc_batch:.4f}')
      eval_loss_this_epoch = eval_loss_this_epoch / len(test_dataset)
      eval_loss_epochs.append(eval_loss_this_epoch)
      if _OUTPUT_DIM.value > 1:  # classification
        eval_accuracy_this_epoch = n_correct_predictions / len(y_test)
        eval_accuracy_epochs.append(eval_accuracy_this_epoch)
    return model, train_loss_epochs, eval_loss_epochs, eval_accuracy_epochs

  # privacy
  epsilon = _EPSILON.value
  delta = _DELTA.value
  num_epochs = _NUM_EPOCHS.value
  criterion = get_loss_fn(_LOSS_NAME.value)
  clip_norm = _CLIP_NORM.value
  # get model
  mlp_layers = (
      [int(l.strip()) for l in _MLP_LAYERS.value] if _MLP_LAYERS.value else None
  )
  if mlp_layers:
    print(f'MLP layers: {mlp_layers}')
  model = get_lin_reg_model(
      input_dim=X_train.shape[1],
      precond_matrix=precond_matrix,
      mlp_layers=mlp_layers,
  )
  # load model
  if _MODEL_LOAD_PATH.value is not None:
    model.load_state_dict(torch.load(_MODEL_LOAD_PATH.value, weights_only=True))
  model = ModuleValidator.fix(model)
  # get noise multiplier
  rdp_acc = RDPAccountant()
  rdp_acc.DEFAULT_ALPHAS = (
      [1 + x / 10.0 for x in range(1, 100)]
      + list(range(12, 101))
      + list(range(100, 1000, 10))
  )
  # print(rdp_acc.DEFAULT_ALPHAS)
  if _EPSILON.value is not None:
    noise_multiplier = get_noise_multiplier(
      target_epsilon=epsilon,
      target_delta=delta,
      epochs=num_epochs,
      sample_rate=train_batch_size / len(X_train),
      epsilon_tolerance=0.01,
      custom_accountant=rdp_acc,
    )
    print(
        f'noise multiplier: {noise_multiplier}, epsilon:'
        f' {rdp_acc.get_epsilon(delta):.4f}'
    )
  else:
    noise_multiplier = 0
    print('non-private setting, no noise!')

  model, optimizer, train_data_loader = get_optimizer(
      model,
      optimizer_name=_OPT_NAME.value,
      learning_rate=_LEARNING_RATE.value,
      epsilon=epsilon,
      data_loader=train_data_loader,
      grad_norm=clip_norm,
      noise_multiplier=noise_multiplier,
  )

  model, train_loss, eval_loss, eval_acc = train(
      model,
      train_data_loader,
      test_data_loader,
      optimizer,
      criterion=criterion,
      num_epochs=num_epochs,
  )

  # write down experiment results
  def get_exp_res_path():
    if _RES_DIR.value is None:
      return None, None
    precond_matrix_str = '_precond' if precond_matrix is not None else '_dpsgd'
    epochs_str = f'_epochs-{num_epochs}'
    learning_rate_str = f'_lr-{_LEARNING_RATE.value}'
    train_batch_size_str = f'_trainb-{train_batch_size}'  # train batch size
    test_batch_size_str = f'_testb-{test_batch_size}'  # aka., eval batch size
    epsilon_str = f'_eps-{epsilon}' if epsilon else ''
    clip_norm_str = f'_clip-{clip_norm}' if clip_norm else ''
    exp_no_str = f'_exp-{_EXP_NO.value}' if _EXP_NO.value is not None else ''
    init_str = (
        f'_initstd-{_INIT_STD.value}' if _INIT_STD.value is not None else ''
    )
    if mlp_layers:
      mlp_layers_str = ','.join([str(l) for l in mlp_layers])
      mlp_layers_str = f'_mlp-{mlp_layers_str}'
    else:
      mlp_layers_str = ''
    exp_str = (
        f'synthetic_lin_reg{epochs_str}{learning_rate_str}'
        f'{train_batch_size_str}{test_batch_size_str}'
        f'{epsilon_str}{clip_norm_str}{mlp_layers_str}{init_str}'
        f'{precond_matrix_str}{exp_no_str}'
    )
    return _RES_DIR.value + '/' + exp_str + '.npy', exp_str

  exp_res_path, exp_str = get_exp_res_path()
  if exp_str and _MODEL_SAVE_PATH.value:
      model_weight_path = _MODEL_SAVE_PATH.value + '/' + exp_str + '.pth'
      torch.save(model.state_dict(), model_weight_path)
  if exp_res_path:
    train_loss = np.array(train_loss)
    eval_loss = np.array(eval_loss)
    if _OUTPUT_DIM.value > 1:  # classification
      eval_acc = np.array(eval_acc)
      all_results = np.vstack((train_loss, eval_loss, eval_acc))
    else:
      all_results = np.vstack((train_loss, eval_loss))
    np.save(exp_res_path, all_results)


if __name__ == '__main__':
  app.run(main)