import numpy as np


def load_synthetic_data(data_dir, sp=None):
  sp_str = '' if sp is None else f'_sp{sp}'
  X_train = np.load(data_dir + f'/X_train{sp_str}.npy')
  y_train = np.load(data_dir + f'/y_train{sp_str}.npy')
  X_test = np.load(data_dir + f'/X_test{sp_str}.npy')
  y_test = np.load(data_dir + f'/y_test{sp_str}.npy')
  return X_train, y_train, X_test, y_test


def load_dataset(data_dir):
  X_train = np.load(data_dir + '/X_train.npy')
  y_train = np.load(data_dir + '/y_train.npy')
  X_test = np.load(data_dir + '/X_test.npy')
  y_test = np.load(data_dir + '/y_test.npy')
  return X_train, y_train, X_test, y_test


def load_precond_matrix(path):
  precond_matrix = np.load(path)
  return precond_matrix
