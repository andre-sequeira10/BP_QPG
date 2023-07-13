import matplotlib.pyplot as plt
import numpy as np
import glob

reward_cumul_prod = []
reward_cumul_true_prod = []
reward_cumul_full = []
reward_cumul_sum = []
reward_prod = []
reward_true_prod= []
reward_full = []
reward_sum= []
var_prod = []
var_true_prod = []
var_full = []
var_sum = []
# Load data
for np_name in glob.glob('/Users/andresequeira/Desktop/bp_qpg_PART_I/data_cartpole_born/cartpole_prod_n_block_diag_NG_grads_norm*'):
    reward_prod.append(np.load(np_name))
for np_name in glob.glob('/Users/andresequeira/Desktop/bp_qpg_PART_I/data_cartpole_born/cartpole_prod_n_block_diag_NG_grads_var*'):
    var_prod.append(np.load(np_name))
for np_name in glob.glob('/Users/andresequeira/Desktop/bp_qpg_PART_I/data_cartpole_born/cartpole_prod_n_block_diag_NG_ -*'):
    reward_cumul_prod.append(np.load(np_name))

for np_name in glob.glob('/Users/andresequeira/Desktop/bp_qpg_PART_I/data_cartpole_born/cartpole_Adam_block_diag_NG_grads_norm*'):
    reward_full.append(np.load(np_name))
for np_name in glob.glob('/Users/andresequeira/Desktop/bp_qpg_PART_I/data_cartpole_born/cartpole_Adam_block_diag_NG_vars*'):
    var_full.append(np.load(np_name))
for np_name in glob.glob('/Users/andresequeira/Desktop/bp_qpg_PART_I/data_cartpole_born/cartpole_Adam_block_diag_NG_ -*'):
    reward_cumul_full.append(np.load(np_name))

for np_name in glob.glob('/Users/andresequeira/Desktop/bp_qpg_PART_I/data_cartpole_born/cartpole_sum_n_5_block_diag_NG_grads_norm*'):
    reward_sum.append(np.load(np_name))
for np_name in glob.glob('/Users/andresequeira/Desktop/bp_qpg_PART_I/data_cartpole_born/cartpole_sum_n_5_block_diag_NG_grads_var*'):
    var_sum.append(np.load(np_name))
for np_name in glob.glob('/Users/andresequeira/Desktop/bp_qpg_PART_I/data_cartpole_born/cartpole_sum_n_5_block_diag_NG_ -*'):
    reward_cumul_sum.append(np.load(np_name))

for np_name in glob.glob('cartpole_true_prod_is_n_5_block_diag_NG_grads_norm*'):
    reward_true_prod.append(np.load(np_name))
for np_name in glob.glob('cartpole_true_prod_is_n_5_block_diag_NG_grads_var*'):
    var_true_prod.append(np.load(np_name))
for np_name in glob.glob('cartpole_true_prod_is_n_5_block_diag_NG_ -*'):
    reward_cumul_true_prod.append(np.load(np_name))


# Calculate means and standard deviations
reward_prod_mean = np.array(reward_prod).mean(axis=0)
reward_prod_std = np.array(reward_prod).std(axis=0)

reward_full_mean = np.array(reward_full).mean(axis=0)
reward_full_std = np.array(reward_full).std(axis=0)

reward_sum_mean = np.array(reward_sum).mean(axis=0)
reward_sum_std = np.array(reward_sum).std(axis=0)

var_prod_mean = np.array(var_prod).mean(axis=0)
var_prod_std = np.array(var_prod).std(axis=0)

var_full_mean = np.array(var_full).mean(axis=0)
var_full_std = np.array(var_full).std(axis=0)

var_sum_mean = np.array(var_sum).mean(axis=0)
var_sum_std = np.array(var_sum).std(axis=0)

reward_cumul_prod_mean = np.array(reward_cumul_prod).mean(axis=0)
reward_cumul_prod_std = np.array(reward_cumul_prod).std(axis=0)

reward_cumul_full.pop(-2)
reward_cumul_full.pop(0)
reward_cumul_full_mean = np.array(reward_cumul_full).mean(axis=0)
reward_cumul_full_std = np.array(reward_cumul_full).std(axis=0)

reward_cumul_sum_mean = np.array(reward_cumul_sum).mean(axis=0)
reward_cumul_sum_std = np.array(reward_cumul_sum).std(axis=0)

reward_cumul_true_prod_mean = np.array(reward_cumul_true_prod).mean(axis=0)
reward_cumul_true_prod_std = np.array(reward_cumul_true_prod).std(axis=0)

var_true_prod_mean = np.array(var_true_prod).mean(axis=0)
var_true_prod_std = np.array(var_true_prod).std(axis=0)

reward_true_prod_mean = np.array(reward_true_prod).mean(axis=0)
reward_true_prod_std = np.array(reward_true_prod).std(axis=0)

# Smoothing
window = 10
smoothed_reward_prod = [np.mean(reward_prod_mean[i-window:i+1]) if i > window 
                        else np.mean(reward_prod_mean[:i+1]) for i in range(len(reward_prod_mean))]
smoothed_reward_full = [np.mean(reward_full_mean[i-window:i+1]) if i > window
                        else np.mean(reward_full_mean[:i+1]) for i in range(len(reward_full_mean))]
smoothed_var_prod = [np.mean(var_prod_mean[i-window:i+1]) if i > window
                     else np.mean(var_prod_mean[:i+1]) for i in range(len(var_prod_mean))]
smoothed_var_full = [np.mean(var_full_mean[i-window:i+1]) if i > window
                     else np.mean(var_full_mean[:i+1]) for i in range(len(var_full_mean))]
smoothed_reward_cumul_prod = [np.mean(reward_cumul_prod_mean[i-window:i+1]) if i > window
                              else np.mean(reward_cumul_prod_mean[:i+1]) for i in range(len(reward_cumul_prod_mean))]
smoothed_reward_cumul_full = [np.mean(reward_cumul_full_mean[i-window:i+1]) if i > window
                              else np.mean(reward_cumul_full_mean[:i+1]) for i in range(len(reward_cumul_full_mean))]
smoothed_reward_cumul_sum = [np.mean(reward_cumul_sum_mean[i-window:i+1]) if i > window
                                else np.mean(reward_cumul_sum_mean[:i+1]) for i in range(len(reward_cumul_sum_mean))]
smoothed_reward_sum = [np.mean(reward_sum_mean[i-window:i+1]) if i > window
                          else np.mean(reward_sum_mean[:i+1]) for i in range(len(reward_sum_mean))]
smoothed_var_sum = [np.mean(var_sum_mean[i-window:i+1]) if i > window
                    else np.mean(var_sum_mean[:i+1]) for i in range(len(var_sum_mean))]
smoothed_reward_true_prod = [np.mean(reward_true_prod_mean[i-window:i+1]) if i > window
                                else np.mean(reward_true_prod_mean[:i+1]) for i in range(len(reward_true_prod_mean))]
smoothed_var_true_prod = [np.mean(var_true_prod_mean[i-window:i+1]) if i > window
                            else np.mean(var_true_prod_mean[:i+1]) for i in range(len(var_true_prod_mean))]
smoothed_reward_cumul_true_prod = [np.mean(reward_cumul_true_prod_mean[i-window:i+1]) if i > window
                                        else np.mean(reward_cumul_true_prod_mean[:i+1]) for i in range(len(reward_cumul_true_prod_mean))]

                            

# Define x for each subplot
x_norm_var = 10*np.arange(min(len(reward_prod_mean), len(reward_full_mean), len(var_prod_mean), len(var_full_mean)))
x_reward = np.arange(min(len(reward_cumul_prod_mean), len(reward_cumul_full_mean)))

fig, axs = plt.subplots(1, 3, figsize=(15, 4))



# First subplot: Policy gradient norm
axs[0].set_xlim([0, 500])
axs[0].fill_between(x_norm_var, reward_prod_mean[:len(x_norm_var)] - reward_prod_std[:len(x_norm_var)], reward_prod_mean[:len(x_norm_var)] + reward_prod_std[:len(x_norm_var)], color="darkblue", alpha=0.1)
axs[0].plot(x_norm_var, smoothed_reward_prod[:len(x_norm_var)], label="prod approx", color="darkblue", alpha=0.7)

axs[0].fill_between(x_norm_var, reward_true_prod_mean[:len(x_norm_var)] - reward_true_prod_std[:len(x_norm_var)], reward_true_prod_mean[:len(x_norm_var)] + reward_true_prod_std[:len(x_norm_var)], color="darkorange", alpha=0.1)
axs[0].plot(x_norm_var, smoothed_reward_true_prod[:len(x_norm_var)], label="prod approx w/ input scaling", color="darkorange", alpha=0.7)

axs[0].fill_between(x_norm_var, reward_full_mean[:len(x_norm_var)] - reward_full_std[:len(x_norm_var)], reward_full_mean[:len(x_norm_var)] + reward_full_std[:len(x_norm_var)], color="purple", alpha=0.3)
axs[0].plot(x_norm_var, smoothed_reward_full[:len(x_norm_var)], label="Born", color="purple", alpha=0.7)

axs[0].fill_between(x_norm_var, reward_sum_mean[:len(x_norm_var)] - reward_sum_std[:len(x_norm_var)], reward_sum_mean[:len(x_norm_var)] + reward_sum_std[:len(x_norm_var)], color="darkgreen", alpha=0.1)
axs[0].plot(x_norm_var, smoothed_reward_sum[:len(x_norm_var)], label="mean approx", color="darkgreen", alpha=0.7)

axs[0].legend()
axs[0].set_ylabel('Policy gradient norm')
axs[0].set_xlabel('Episodes')

# Second subplot: Policy gradient variance
axs[1].fill_between(x_norm_var, var_prod_mean[:len(x_norm_var)] - var_prod_std[:len(x_norm_var)], var_prod_mean[:len(x_norm_var)] + var_prod_std[:len(x_norm_var)], color="darkblue", alpha=0.1)
axs[1].plot(x_norm_var, smoothed_var_prod[:len(x_norm_var)], label="prod approx", color="darkblue", alpha=0.7)

axs[1].set_xlim([0, 500])
axs[1].fill_between(x_norm_var, var_full_mean[:len(x_norm_var)] - var_full_std[:len(x_norm_var)], var_full_mean[:len(x_norm_var)] + var_full_std[:len(x_norm_var)], color="purple", alpha=0.3)
axs[1].plot(x_norm_var, smoothed_var_full[:len(x_norm_var)], label="Born", color="purple", alpha=0.7)
axs[1].fill_between(x_norm_var, var_sum_mean[:len(x_norm_var)] - var_sum_std[:len(x_norm_var)], var_sum_mean[:len(x_norm_var)] + var_sum_std[:len(x_norm_var)], color="darkgreen", alpha=0.1)
axs[1].plot(x_norm_var, smoothed_var_sum[:len(x_norm_var)], label="mean approx", color="darkgreen", alpha=0.7)

axs[1].fill_between(x_norm_var, var_true_prod_mean[:len(x_norm_var)] - var_true_prod_std[:len(x_norm_var)], var_true_prod_mean[:len(x_norm_var)] + var_true_prod_std[:len(x_norm_var)], color="darkorange", alpha=0.1)
axs[1].plot(x_norm_var, smoothed_var_true_prod[:len(x_norm_var)], label="prod approx w/ input scaling", color="darkorange", alpha=0.7)

axs[1].legend()
axs[1].set_ylabel('Policy gradient variance')
axs[1].set_xlabel('Episodes')

# Third subplot: Cumulative rewards
axs[2].set_ylim([0, 200])
axs[2].set_xlim([0, 500])
axs[2].fill_between(x_reward, reward_cumul_prod_mean[:len(x_reward)] - reward_cumul_prod_std[:len(x_reward)], reward_cumul_prod_mean[:len(x_reward)] + reward_cumul_prod_std[:len(x_reward)], color="darkblue", alpha=0.2)
axs[2].plot(x_reward, smoothed_reward_cumul_prod[:len(x_reward)], label="prod approx", color="darkblue", alpha=0.7)

axs[2].fill_between(x_reward, reward_cumul_full_mean[:len(x_reward)] - reward_cumul_full_std[:len(x_reward)], reward_cumul_full_mean[:len(x_reward)] + reward_cumul_full_std[:len(x_reward)], color="purple", alpha=0.1)
axs[2].plot(x_reward, smoothed_reward_cumul_full[:len(x_reward)], label="Born", color="purple", alpha=0.7)
axs[2].fill_between(x_reward, reward_cumul_sum_mean[:len(x_reward)] - reward_cumul_sum_std[:len(x_reward)], reward_cumul_sum_mean[:len(x_reward)] + reward_cumul_sum_std[:len(x_reward)], color="darkgreen", alpha=0.1)
axs[2].plot(x_reward, smoothed_reward_cumul_sum[:len(x_reward)], label="mean approx", color="darkgreen", alpha=0.7)

axs[2].fill_between(x_reward, reward_cumul_true_prod_mean[:len(x_reward)] - reward_cumul_true_prod_std[:len(x_reward)], reward_cumul_true_prod_mean[:len(x_reward)] + reward_cumul_true_prod_std[:len(x_reward)], color="darkorange", alpha=0.1)
axs[2].plot(x_reward, smoothed_reward_cumul_true_prod[:len(x_reward)], label="prod approx w/ input scaling", color="darkorange", alpha=0.7)

axs[2].legend()
axs[2].set_ylabel('Cumulative rewards')
axs[2].set_xlabel('Episodes')
axs[2].legend(loc='lower right', bbox_to_anchor=(1.0, 0.0))
plt.tight_layout()  # Adjust the spacing between subplots
plt.rcParams['figure.dpi'] = 1200
plt.rcParams['savefig.dpi'] = 1200

plt.show()
