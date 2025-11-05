import os
import numpy as np
import scipy.io as sio # This will be used to load an MATLAB file
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as bpdf
from self_fun_py.HW8Fun import produce_trun_mean_cov, plot_trunc_mean, plot_trunc_cov

bp_low = 0.5
bp_upp = 6
electrode_num = 16
electrode_name_ls = ['F3', 'Fz', 'F4', 'T7', 'C3', 'Cz', 'C4', 'T8', 'CP3', 'CP4', 'P3', 'Pz', 'P4', 'PO7', 'PO8', 'Oz']

parent_dir = '/Users/isaachuntington/Documents/GitHub/BIOS-584'
parent_data_dir = '{}/data'.format(parent_dir)
time_index = np.linspace(0, 800, 25) # This is a hypothetic time range up to 800 ms after each stimulus.

subject_name = 'K114'
session_name = '001_BCI_TRN'

new_dataset='{}/K114_001_BCI_TRN_Truncated_Data_0.5_6.mat'.format(parent_data_dir)
eeg_trunc_obj=sio.loadmat(new_dataset)

print(eeg_trunc_obj.keys())
Signal = eeg_trunc_obj['Signal']
Type = eeg_trunc_obj['Type']
eeg_trunc_signal = Signal
eeg_trunc_type = Type
print(eeg_trunc_signal.shape)
print(eeg_trunc_type.shape)
eeg_trunc_type = np.squeeze(eeg_trunc_type, axis=-1)
print(eeg_trunc_type.shape)
print(eeg_trunc_type)
print(np.unique(eeg_trunc_type))

input_signal=eeg_trunc_signal
input_type=eeg_trunc_type
E_val=16
feature_len=input_signal.shape[1]
lpe=feature_len//E_val


my_output1 = produce_trun_mean_cov(eeg_trunc_signal, eeg_trunc_type, electrode_num)
my_output1

plot_trunc_mean(my_output1[0], my_output1[1], subject_name, time_index, E_val, electrode_name_ls)

plot_trunc_cov(my_output1[2], "Target", time_index, subject_name, E_val, electrode_name_ls)
plot_trunc_cov(my_output1[3], "Non Target", time_index, subject_name, E_val, electrode_name_ls)
plot_trunc_cov(my_output1[4], "All", time_index, subject_name, E_val, electrode_name_ls)