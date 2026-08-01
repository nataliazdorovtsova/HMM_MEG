% This script will implement OHBA's preprocessing pipeline to resting-state
% data from RED that has been 38 parcellated

function output = finish_preprocessing_meg(sub)

outputdir = '/imaging/astle/users/nz01/RED_MEG/Outputs_FullDataset/Preproc_Outputs3'; % where outputs should go

setenv('FSLDIR','/imaging/local/software/fsl/v5.0.7/x86_64/fsl/'); % FSL
addpath(genpath('/imaging/astle/users/nz01/RED_MEG/Scripts/HMM-MAR-master/')); % scripts
addpath(genpath('/imaging/astle/users/nz01/RED_MEG/Scripts/SPM_Files')); % data directory
addpath(genpath('/imaging/local/software/spm_cbu_svn/releases/spm12_latest/')); % SPM v12
addpath(genpath('/imaging/local/software/spm_toolbox/osl/')); % ROI nets for leakage 

% 52 participants
data =  {'/imaging/astle/users/nz01/RED_MEG/Scripts/SPM_Files/bffilteredspmeeg_99003_cleanica_sss'
         '/imaging/astle/users/nz01/RED_MEG/Scripts/SPM_Files/bffilteredspmeeg_99009_cleanica_sss'
         '/imaging/astle/users/nz01/RED_MEG/Scripts/SPM_Files/bffilteredspmeeg_99011_cleanica_sss'
         '/imaging/astle/users/nz01/RED_MEG/Scripts/SPM_Files/bffilteredspmeeg_99013_cleanica_sss'
         '/imaging/astle/users/nz01/RED_MEG/Scripts/SPM_Files/bffilteredspmeeg_99015_cleanica_sss'
         '/imaging/astle/users/nz01/RED_MEG/Scripts/SPM_Files/bffilteredspmeeg_99016_cleanica_sss'
         '/imaging/astle/users/nz01/RED_MEG/Scripts/SPM_Files/bffilteredspmeeg_99018_cleanica_sss'
         '/imaging/astle/users/nz01/RED_MEG/Scripts/SPM_Files/bffilteredspmeeg_99019_cleanica_sss'
         '/imaging/astle/users/nz01/RED_MEG/Scripts/SPM_Files/bffilteredspmeeg_99020_cleanica_sss'
         '/imaging/astle/users/nz01/RED_MEG/Scripts/SPM_Files/bffilteredspmeeg_99023_cleanica_sss'
         '/imaging/astle/users/nz01/RED_MEG/Scripts/SPM_Files/bffilteredspmeeg_99025_cleanica_sss'
         '/imaging/astle/users/nz01/RED_MEG/Scripts/SPM_Files/bffilteredspmeeg_99026_cleanica_sss'
         '/imaging/astle/users/nz01/RED_MEG/Scripts/SPM_Files/bffilteredspmeeg_99027_cleanica_sss'
         '/imaging/astle/users/nz01/RED_MEG/Scripts/SPM_Files/bffilteredspmeeg_99028_cleanica_sss'
         '/imaging/astle/users/nz01/RED_MEG/Scripts/SPM_Files/bffilteredspmeeg_99030_cleanica_sss'
         '/imaging/astle/users/nz01/RED_MEG/Scripts/SPM_Files/bffilteredspmeeg_99034_cleanica_sss'
         '/imaging/astle/users/nz01/RED_MEG/Scripts/SPM_Files/bffilteredspmeeg_99038_cleanica_sss'
         '/imaging/astle/users/nz01/RED_MEG/Scripts/SPM_Files/bffilteredspmeeg_99040_cleanica_sss'
         '/imaging/astle/users/nz01/RED_MEG/Scripts/SPM_Files/bffilteredspmeeg_99044_cleanica_sss'
         '/imaging/astle/users/nz01/RED_MEG/Scripts/SPM_Files/bffilteredspmeeg_99047_cleanica_sss'
         '/imaging/astle/users/nz01/RED_MEG/Scripts/SPM_Files/bffilteredspmeeg_99058_cleanica_sss'
         '/imaging/astle/users/nz01/RED_MEG/Scripts/SPM_Files/bffilteredspmeeg_99059_cleanica_sss'
         '/imaging/astle/users/nz01/RED_MEG/Scripts/SPM_Files/bffilteredspmeeg_99060_cleanica_sss'
         '/imaging/astle/users/nz01/RED_MEG/Scripts/SPM_Files/bffilteredspmeeg_99063_cleanica_sss'
         '/imaging/astle/users/nz01/RED_MEG/Scripts/SPM_Files/bffilteredspmeeg_99064_cleanica_sss'
         '/imaging/astle/users/nz01/RED_MEG/Scripts/SPM_Files/bffilteredspmeeg_99070_cleanica_sss'
         '/imaging/astle/users/nz01/RED_MEG/Scripts/SPM_Files/bffilteredspmeeg_99072_cleanica_sss'
         '/imaging/astle/users/nz01/RED_MEG/Scripts/SPM_Files/bffilteredspmeeg_99073_cleanica_sss'
         '/imaging/astle/users/nz01/RED_MEG/Scripts/SPM_Files/bffilteredspmeeg_99078_cleanica_sss'
         '/imaging/astle/users/nz01/RED_MEG/Scripts/SPM_Files/bffilteredspmeeg_99080_cleanica_sss'
         '/imaging/astle/users/nz01/RED_MEG/Scripts/SPM_Files/bffilteredspmeeg_99086_cleanica_sss'
         '/imaging/astle/users/nz01/RED_MEG/Scripts/SPM_Files/bffilteredspmeeg_99089_cleanica_sss'
         '/imaging/astle/users/nz01/RED_MEG/Scripts/SPM_Files/bffilteredspmeeg_99090_cleanica_sss'
         '/imaging/astle/users/nz01/RED_MEG/Scripts/SPM_Files/bffilteredspmeeg_99091_cleanica_sss'
         '/imaging/astle/users/nz01/RED_MEG/Scripts/SPM_Files/bffilteredspmeeg_99092_cleanica_sss'
         '/imaging/astle/users/nz01/RED_MEG/Scripts/SPM_Files/bffilteredspmeeg_99094_cleanica_sss'
         '/imaging/astle/users/nz01/RED_MEG/Scripts/SPM_Files/bffilteredspmeeg_99099_cleanica_sss'
         '/imaging/astle/users/nz01/RED_MEG/Scripts/SPM_Files/bffilteredspmeeg_99103_cleanica_sss'
         '/imaging/astle/users/nz01/RED_MEG/Scripts/SPM_Files/bffilteredspmeeg_99112_cleanica_sss'
         '/imaging/astle/users/nz01/RED_MEG/Scripts/SPM_Files/bffilteredspmeeg_99114_cleanica_sss'
         '/imaging/astle/users/nz01/RED_MEG/Scripts/SPM_Files/bffilteredspmeeg_99119_cleanica_sss'
         '/imaging/astle/users/nz01/RED_MEG/Scripts/SPM_Files/bffilteredspmeeg_99125_cleanica_sss'
         '/imaging/astle/users/nz01/RED_MEG/Scripts/SPM_Files/bffilteredspmeeg_99128_cleanica_sss'
         '/imaging/astle/users/nz01/RED_MEG/Scripts/SPM_Files/bffilteredspmeeg_99129_cleanica_sss'
         '/imaging/astle/users/nz01/RED_MEG/Scripts/SPM_Files/bffilteredspmeeg_99134_cleanica_sss'
         '/imaging/astle/users/nz01/RED_MEG/Scripts/SPM_Files/bffilteredspmeeg_99146_cleanica_sss'
         '/imaging/astle/users/nz01/RED_MEG/Scripts/SPM_Files/bffilteredspmeeg_calm01_cleanica_sss'
         '/imaging/astle/users/nz01/RED_MEG/Scripts/SPM_Files/bffilteredspmeeg_calm02_cleanica_sss'
         '/imaging/astle/users/nz01/RED_MEG/Scripts/SPM_Files/bffilteredspmeeg_calm03_cleanica_sss'
         '/imaging/astle/users/nz01/RED_MEG/Scripts/SPM_Files/bffilteredspmeeg_calm04_cleanica_sss'
         '/imaging/astle/users/nz01/RED_MEG/Scripts/SPM_Files/bffilteredspmeeg_calm05_cleanica_sss'
         '/imaging/astle/users/nz01/RED_MEG/Scripts/SPM_Files/bffilteredspmeeg_calm06_cleanica_sss'}; % number of subjects x file location for each subject

Tc = csvread('T_Full.csv');
T = {[Tc(1)],[Tc(2)],[Tc(3)],[Tc(4)],[Tc(5)],...
    [Tc(6)],[Tc(7)],[Tc(8)],[Tc(9)],[Tc(10)],...
    [Tc(11)],[Tc(12)],[Tc(13)],[Tc(14)],[Tc(15)],...
    [Tc(16)],[Tc(17)],[Tc(18)],[Tc(19)],[Tc(20)],...
    [Tc(21)],[Tc(22)],[Tc(23)],[Tc(24)],[Tc(25)],...
    [Tc(26)],[Tc(27)],[Tc(28)],[Tc(29)],[Tc(30)],...
    [Tc(31)],[Tc(32)],[Tc(33)],[Tc(34)],[Tc(35)],...
    [Tc(36)],[Tc(37)],[Tc(38)],[Tc(39)],[Tc(40)],...
    [Tc(41)],[Tc(42)],[Tc(43)],[Tc(44)],[Tc(45)],...
    [Tc(46)],[Tc(47)],[Tc(48)],[Tc(49)],[Tc(50)],...
    [Tc(51)],[Tc(52)]}; % in the square brackets, we specify the number of time points for each subject

% get the ids of the participants
labels = {};
for j = 1:length(data);
    labels{j} = data{j}(end-18:end-13);
end

% take this subjects data, T and label
label = labels{sub};

 %% Options
 
 options = struct();
 options.filter = []; % we have filtered already in SPM (1-30Hz)
 options.detrend = 1; % detrend
 options.standardise = 1; % standardise (center and make the standard deviation equal to one) 
                           % the signal for each trial, such that all trials and channels have 
                           % the same mean and standard deviation
 options.leakagecorr = -1; % solution to signal leakage are implemented with -1: the symmetric 
                          % orthogonalisation proposed by Colclough et al. (2015) 
 options.onpower = 1; % computing the Hilbert envelope of the signal to represent power data and not raw data
 %options.pca = 50; % we would specify if we wanted pca here; we don't,
 %since we have already applied a parcellation via PCA
 options.varimax = 0; % no rotation
 options.Fs = 1000; % sampling frequency was originally 1000Hz
 options.downsample = 250; % we want to downsample our data to 250Hz
 
 % HMM options
 
options.K = 7; % number of states
options.timelag = 1;
options.orderoffset = 0;
options.verbose = 1; 
options.order = 0; % maximum order of the MAR model; if zero, an HMM with 
                   % Gaussian observations is trained (mandatory, with no default).
options.zeromean = 0; % if 1, the mean of the time series will not be used to drive 
                      % the states (default to 1 if order is higher than 0, and 0 otherwise).

 %% Preprocess

 [data,T,options] = hmmpreprocess(data(sub),T(sub),options);
 
 % saving directory
 cd(outputdir);
 save(sprintf('data_%s.mat',label),'data','-v7.3'); % data output
 save(sprintf('T_%s.mat',label),'T','-v7.3'); % time output
 save('options.mat','options','-v7.3'); % options output 
 
end