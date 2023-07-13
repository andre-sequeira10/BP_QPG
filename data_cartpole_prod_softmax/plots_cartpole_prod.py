import matplotlib.pyplot as plt
import numpy as np
import glob

reward_cumul_global = []
reward_cumul_local = []
reward_cumul_local_input_scaling = []
reward_cumul_global_input_scaling = []

grad_norm_global = []
grad_norm_local = []
grad_norm_local_input_scaling = []
grad_norm_global_input_scaling = []

var_global = []
var_local = []
var_local_input_scaling = []
var_global_input_scaling = []

# Load data
for np_name in glob.glob('cartpole_0_inpt_scaling-0_n_layers-3*_grads_norm*'):
    grad_norm_global.append(np.load(np_name))
for np_name in glob.glob('cartpole_0_inpt_scaling-0_n_layers-3*grads_var*'):
    var_global.append(np.load(np_name))
for np_name in glob.glob('cartpole_0_inpt_scaling-0_n_layers-3||*'):
    reward_cumul_global.append(np.load(np_name))

for np_name in glob.glob('cartpole_1_inpt_scaling-0_n_layers-3*_grads_norm*'):
    grad_norm_local.append(np.load(np_name))
for np_name in glob.glob('cartpole_1_inpt_scaling-0_n_layers-3*grads_var*'):
    var_local.append(np.load(np_name))
for np_name in glob.glob('cartpole_1_inpt_scaling-0_n_layers-3||*'):
    reward_cumul_local.append(np.load(np_name))

for np_name in glob.glob('cartpole_1_inpt_scaling-1_n_layers-3*_grads_norm*'):
    grad_norm_local_input_scaling.append(np.load(np_name))
for np_name in glob.glob('cartpole_1_inpt_scaling-1_n_layers-3*grads_var*'):
    var_local_input_scaling.append(np.load(np_name))
for np_name in glob.glob('cartpole_1_inpt_scaling-1_n_layers-3||*'):
    reward_cumul_local_input_scaling.append(np.load(np_name))

for np_name in glob.glob('cartpole_0_inpt_scaling-1_n_layers-3*_grads_norm*'):
    grad_norm_global_input_scaling.append(np.load(np_name))
for np_name in glob.glob('cartpole_0_inpt_scaling-1_n_layers-3*grads_var*'):
    var_global_input_scaling.append(np.load(np_name))
for np_name in glob.glob('cartpole_0_inpt_scaling-1_n_layers-3||*'):
    reward_cumul_global_input_scaling.append(np.load(np_name))

# Calculate means and standard deviations
grad_norm_global_mean = np.array(grad_norm_global).mean(axis=0)
grad_norm_global_std = np.array(grad_norm_global).std(axis=0)

grad_norm_local_mean = np.array(grad_norm_local).mean(axis=0)
grad_norm_local_std = np.array(grad_norm_local).std(axis=0)

var_global_mean = np.array(var_global).mean(axis=0)
var_global_std = np.array(var_global).std(axis=0)

var_local_mean = np.array(var_local).mean(axis=0)
var_local_std = np.array(var_local).std(axis=0)

reward_cumul_global_mean = np.array(reward_cumul_global).mean(axis=0)
reward_cumul_global_std = np.array(reward_cumul_global).std(axis=0)

reward_cumul_local_mean = np.array(reward_cumul_local).mean(axis=0)
reward_cumul_local_std = np.array(reward_cumul_local).std(axis=0)

grad_norm_local_input_scaling_mean = np.array(grad_norm_local_input_scaling).mean(axis=0)
grad_norm_local_input_scaling_std = np.array(grad_norm_local_input_scaling).std(axis=0)

var_local_input_scaling_mean = np.array(var_local_input_scaling).mean(axis=0)
var_local_input_scaling_std = np.array(var_local_input_scaling).std(axis=0)

reward_cumul_local_input_scaling_mean = np.array(reward_cumul_local_input_scaling).mean(axis=0)
reward_cumul_local_input_scaling_std = np.array(reward_cumul_local_input_scaling).std(axis=0)

grad_norm_global_input_scaling_mean = np.array(grad_norm_global_input_scaling).mean(axis=0)
grad_norm_global_input_scaling_std = np.array(grad_norm_global_input_scaling).std(axis=0)

var_global_input_scaling_mean = np.array(var_global_input_scaling).mean(axis=0)
var_global_input_scaling_std = np.array(var_global_input_scaling).std(axis=0)

reward_cumul_global_input_scaling_mean = np.array(reward_cumul_global_input_scaling).mean(axis=0)
reward_cumul_global_input_scaling_std = np.array(reward_cumul_global_input_scaling).std(axis=0)

# Smoothing
window = 10
smoothed_grad_norm_global = [np.mean(grad_norm_global_mean[i-window:i+1]) if i > window 
                        else np.mean(grad_norm_global_mean[:i+1]) for i in range(len(grad_norm_global_mean))]
smoothed_grad_norm_local = [np.mean(grad_norm_local_mean[i-window:i+1]) if i > window
                        else np.mean(grad_norm_local_mean[:i+1]) for i in range(len(grad_norm_local_mean))]
smoothed_var_global = [np.mean(var_global_mean[i-window:i+1]) if i > window
                        else np.mean(var_global_mean[:i+1]) for i in range(len(var_global_mean))]
smoothed_var_local = [np.mean(var_local_mean[i-window:i+1]) if i > window
                        else np.mean(var_local_mean[:i+1]) for i in range(len(var_local_mean))]
smoothed_reward_cumul_global = [np.mean(reward_cumul_global_mean[i-window:i+1]) if i > window
                        else np.mean(reward_cumul_global_mean[:i+1]) for i in range(len(reward_cumul_global_mean))]
smoothed_reward_cumul_local = [np.mean(reward_cumul_local_mean[i-window:i+1]) if i > window
                        else np.mean(reward_cumul_local_mean[:i+1]) for i in range(len(reward_cumul_local_mean))]
smoothed_grad_norm_local_input_scaling = [np.mean(grad_norm_local_input_scaling_mean[i-window:i+1]) if i > window
                        else np.mean(grad_norm_local_input_scaling_mean[:i+1]) for i in range(len(grad_norm_local_input_scaling_mean))]
smoothed_var_local_input_scaling = [np.mean(var_local_input_scaling_mean[i-window:i+1]) if i > window
                        else np.mean(var_local_input_scaling_mean[:i+1]) for i in range(len(var_local_input_scaling_mean))]
smoothed_reward_cumul_local_input_scaling = [np.mean(reward_cumul_local_input_scaling_mean[i-window:i+1]) if i > window
                        else np.mean(reward_cumul_local_input_scaling_mean[:i+1]) for i in range(len(reward_cumul_local_input_scaling_mean))]
smoothed_grad_norm_global_input_scaling = [np.mean(grad_norm_global_input_scaling_mean[i-window:i+1]) if i > window
                        else np.mean(grad_norm_global_input_scaling_mean[:i+1]) for i in range(len(grad_norm_global_input_scaling_mean))]
smoothed_var_global_input_scaling = [np.mean(var_global_input_scaling_mean[i-window:i+1]) if i > window
                        else np.mean(var_global_input_scaling_mean[:i+1]) for i in range(len(var_global_input_scaling_mean))]
smoothed_reward_cumul_global_input_scaling = [np.mean(reward_cumul_global_input_scaling_mean[i-window:i+1]) if i > window
                        else np.mean(reward_cumul_global_input_scaling_mean[:i+1]) for i in range(len(reward_cumul_global_input_scaling_mean))]

    

# Define x for each subplot
x_norm_var = 10*np.arange(min(len(grad_norm_global_mean), len(grad_norm_local_mean), len(grad_norm_local_input_scaling_mean), len(grad_norm_global_input_scaling_mean), len(var_global_mean), len(var_local_mean), len(var_local_input_scaling_mean), len(var_global_input_scaling_mean)))
x_reward = np.arange(min(len(reward_cumul_global_mean), len(reward_cumul_local_mean), len(reward_cumul_local_input_scaling_mean), len(reward_cumul_global_input_scaling_mean)))

fig, axs = plt.subplots(1, 3, figsize=(15, 4))



# First subplot: Policy gradient norm
axs[0].set_xlim([0, 500])
axs[0].fill_between(x_norm_var, grad_norm_global_mean[:len(x_norm_var)] - grad_norm_global_std[:len(x_norm_var)], grad_norm_global_mean[:len(x_norm_var)] + grad_norm_global_std[:len(x_norm_var)], color="darkblue", alpha=0.1)
axs[0].plot(x_norm_var, smoothed_grad_norm_global[:len(x_norm_var)], label="global", color="darkblue", alpha=0.7)

axs[0].fill_between(x_norm_var, grad_norm_local_mean[:len(x_norm_var)] - grad_norm_local_std[:len(x_norm_var)], grad_norm_local_mean[:len(x_norm_var)] + grad_norm_local_std[:len(x_norm_var)], color="purple", alpha=0.1)
axs[0].plot(x_norm_var, smoothed_grad_norm_local[:len(x_norm_var)], label="local", color="purple", alpha=0.7)

axs[0].fill_between(x_norm_var, grad_norm_local_input_scaling_mean[:len(x_norm_var)] - grad_norm_local_input_scaling_std[:len(x_norm_var)], grad_norm_local_input_scaling_mean[:len(x_norm_var)] + grad_norm_local_input_scaling_std[:len(x_norm_var)], color="darkorange", alpha=0.1)
axs[0].plot(x_norm_var, smoothed_grad_norm_local_input_scaling[:len(x_norm_var)], label="local input scaling", color="darkorange", alpha=0.7)

axs[0].fill_between(x_norm_var, grad_norm_global_input_scaling_mean[:len(x_norm_var)] - grad_norm_global_input_scaling_std[:len(x_norm_var)], grad_norm_global_input_scaling_mean[:len(x_norm_var)] + grad_norm_global_input_scaling_std[:len(x_norm_var)], color="darkgreen", alpha=0.1)
axs[0].plot(x_norm_var, smoothed_grad_norm_global_input_scaling[:len(x_norm_var)], label="global input scaling", color="darkgreen", alpha=0.7)

axs[0].legend()
axs[0].set_ylabel('Policy gradient norm')
axs[0].set_xlabel('Episodes')

# Second subplot: Policy gradient variance
axs[1].fill_between(x_norm_var, var_global_mean[:len(x_norm_var)] - var_global_std[:len(x_norm_var)], var_global_mean[:len(x_norm_var)] + var_global_std[:len(x_norm_var)], color="darkblue", alpha=0.1)
axs[1].plot(x_norm_var, smoothed_var_global[:len(x_norm_var)], label="global", color="darkblue", alpha=0.7)

axs[1].set_xlim([0, 500])
axs[1].fill_between(x_norm_var, var_local_mean[:len(x_norm_var)] - var_local_std[:len(x_norm_var)], var_local_mean[:len(x_norm_var)] + var_local_std[:len(x_norm_var)], color="purple", alpha=0.3)
axs[1].plot(x_norm_var, smoothed_var_local[:len(x_norm_var)], label="local", color="purple", alpha=0.7)
axs[1].fill_between(x_norm_var, var_local_input_scaling_mean[:len(x_norm_var)] - var_local_input_scaling_std[:len(x_norm_var)], var_local_input_scaling_mean[:len(x_norm_var)] + var_local_input_scaling_std[:len(x_norm_var)], color="darkorange", alpha=0.1)
axs[1].plot(x_norm_var, smoothed_var_local_input_scaling[:len(x_norm_var)], label="local input scaling", color="darkorange", alpha=0.7)
axs[1].fill_between(x_norm_var, var_global_input_scaling_mean[:len(x_norm_var)] - var_global_input_scaling_std[:len(x_norm_var)], var_global_input_scaling_mean[:len(x_norm_var)] + var_global_input_scaling_std[:len(x_norm_var)], color="darkgreen", alpha=0.1)
axs[1].plot(x_norm_var, smoothed_var_global_input_scaling[:len(x_norm_var)], label="global input scaling", color="darkgreen", alpha=0.7)


axs[1].legend()
axs[1].set_ylabel('Policy gradient variance')
axs[1].set_xlabel('Episodes')
axs[1].set_ylim([0, 0.50])

# Third subplot: Cumulative rewards
axs[2].set_ylim([0, 200])
axs[2].set_xlim([0, 500])

axs[2].fill_between(x_reward, reward_cumul_global_mean[:len(x_reward)] - reward_cumul_global_std[:len(x_reward)], reward_cumul_global_mean[:len(x_reward)] + reward_cumul_global_std[:len(x_reward)], color="darkblue", alpha=0.2)
axs[2].plot(x_reward, smoothed_reward_cumul_global[:len(x_reward)], label="global", color="darkblue", alpha=0.7)

axs[2].fill_between(x_reward, reward_cumul_local_mean[:len(x_reward)] - reward_cumul_local_std[:len(x_reward)], reward_cumul_local_mean[:len(x_reward)] + reward_cumul_local_std[:len(x_reward)], color="purple", alpha=0.1)
axs[2].plot(x_reward, smoothed_reward_cumul_local[:len(x_reward)], label="local", color="purple", alpha=0.7)

axs[2].fill_between(x_reward, reward_cumul_local_input_scaling_mean[:len(x_reward)] - reward_cumul_local_input_scaling_std[:len(x_reward)], reward_cumul_local_input_scaling_mean[:len(x_reward)] + reward_cumul_local_input_scaling_std[:len(x_reward)], color="darkorange", alpha=0.1)
axs[2].plot(x_reward, smoothed_reward_cumul_local_input_scaling[:len(x_reward)], label="local input scaling", color="darkorange", alpha=0.7)
axs[2].fill_between(x_reward, reward_cumul_global_input_scaling_mean[:len(x_reward)] - reward_cumul_global_input_scaling_std[:len(x_reward)], reward_cumul_global_input_scaling_mean[:len(x_reward)] + reward_cumul_global_input_scaling_std[:len(x_reward)], color="darkgreen", alpha=0.1)
axs[2].plot(x_reward, smoothed_reward_cumul_global_input_scaling[:len(x_reward)], label="global input scaling", color="darkgreen", alpha=0.7)

axs[2].legend()
axs[2].set_ylabel('Cumulative rewards')
axs[2].set_xlabel('Episodes')
axs[2].legend(loc='lower right', bbox_to_anchor=(1.0, 0.0))
plt.tight_layout()  # Adjust the spacing between subplots
plt.rcParams['figure.dpi'] = 1200
plt.rcParams['savefig.dpi'] = 1200

plt.show()
