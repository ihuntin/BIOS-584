import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as bpdf
import os
os.makedirs("K114", exist_ok=True)

def produce_trun_mean_cov(input_signal, input_type, E_val):
    nsample, feature_len = input_signal.shape
    lpe = feature_len // E_val

    signal_tar = input_signal[input_type == 1, :]
    signal_ntar = input_signal[input_type == 0, :]
    signal_tar_3d = signal_tar.reshape(-1, E_val, lpe) if signal_tar.size > 0 else np.zeros((0, E_val, lpe))
    signal_ntar_3d = signal_ntar.reshape(-1, E_val, lpe) if signal_ntar.size > 0 else np.zeros((0, E_val, lpe))
    signal_all_3d = input_signal.reshape(-1, E_val, lpe)

    signal_tar_mean = np.mean(signal_tar_3d, axis=0) if signal_tar_3d.size > 0 else np.zeros((E_val, lpe))
    signal_ntar_mean = np.mean(signal_ntar_3d, axis=0) if signal_ntar_3d.size > 0 else np.zeros((E_val, lpe))

    signal_tar_cov = np.zeros((E_val, lpe, lpe))
    signal_ntar_cov = np.zeros((E_val, lpe, lpe))
    signal_all_cov = np.zeros((E_val, lpe, lpe))

    for e in range(E_val):
        if signal_tar_3d.size > 0:
            signal_tar_cov[e, :, :] = np.cov(signal_tar_3d[e, :, :], rowvar=False)
        if signal_ntar_3d.size > 0:
            signal_ntar_cov[e, :, :] = np.cov(signal_ntar_3d[e, :, :], rowvar=False)
        signal_all_cov[e, :, :] = np.cov(signal_all_3d[e, :, :], rowvar=False)

    return [signal_tar_mean, signal_ntar_mean, signal_tar_cov, signal_ntar_cov, signal_all_cov]


def plot_trunc_mean(
        eeg_tar_mean, eeg_ntar_mean, subject_name, time_index, E_val, electrode_name_ls,
        y_limit=np.array([-5, 8]), fig_size=(12, 12)
):
    fig, axes = plt.subplots(4, 4, figsize=fig_size)
    fig.suptitle(f"Subject: {subject_name} —Target vs Non-Target ERPs")

    for i in range(E_val):
        ax = axes[i // 4, i % 4]
        ax.plot(time_index, eeg_tar_mean[i, :], label='Target')
        ax.plot(time_index, eeg_ntar_mean[i, :], label='Not Target')
        ax.set_title(electrode_name_ls[i])
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude (muV)")
        ax.set_ylim(y_limit)
        ax.set_title(electrode_name_ls[i])
        if i == 0:
            ax.legend()

    plt.tight_layout()
    plt.show()
    save_path1 = f"K114/Mean.png"
    plt.savefig(save_path1)
    plt.close()


def plot_trunc_cov(
        eeg_cov, cov_type, time_index, subject_name, E_val, electrode_name_ls, fig_size=(14, 12)
):
    x, y = np.meshgrid(time_index, time_index)

    fig, axes = plt.subplots(4, 4, figsize=fig_size)
    fig.suptitle(f"{cov_type} — Subject: {subject_name}", fontsize=16)

    for i in range(E_val):
        ax = axes[i // 4, i % 4]
        cov_matrix = eeg_cov[i, :, :]

        contour = ax.contourf(x, y, cov_matrix, cmap='viridis')
        cbar = plt.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        ax.set_title(electrode_name_ls[i])
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Time (ms)")
        ax.invert_yaxis()

    plt.tight_layout()
    plt.show()
    save_path2=f"K114/covariance_{cov_type}.png"
    plt.savefig(save_path2)
    plt.close()