import os

# HPs
data_dir = 'wine'
cond_matrix_path = 'C.npy'
num_epochs = 128
epsilon = 1

learning_rate = 0.1
clip_norm = 0.1
init_std = 0.01

res_dir = 'dpadam_wine_results'
if not os.path.isdir(res_dir):
    os.mkdir(res_dir)

cmd = f'python3 private_lin_reg_v2.py --data_dir {data_dir} --num_epochs {num_epochs} --epsilon {epsilon} ' \
      f'--learning_rate {learning_rate} --clip_norm {clip_norm} --exp_no 0 --res_dir {res_dir} ' \
      f'--init_std {init_std} --precond_matrix_path {cond_matrix_path} --opt_name adam'
os.system(cmd)


