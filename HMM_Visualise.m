%% Visualise state maps using OSL

% Get SPM running in terminal
addpath('/imaging/astle/users/nz01/RED_MEG/Outputs_FullDataset/HMM_Outputs3/'); % data directory
addpath('/imaging/local/software/freesurfer/latest/x86_64'); % freesurfer
addpath('/imaging/local/software/workbench/v1.3.2/'); % workbench
setenv('OSLDIR','/imaging/local/software/spm_toolbox/osl/'); % OSL directory
osl_startup('/imaging/local/software/spm_toolbox/osl','shared') % startup OSL directory

%% Visualise

osl_render4D('K6.nii.gz','interptype','trilinear','visualise',true)



