%% Spectral Estimation

% Load data and set paths
setenv('OSLDIR','/imaging/local/software/spm_toolbox/osl/'); % OSL
addpath('/imaging/local/software/spm_toolbox/osl/HMM-MAR/'); % scripts
addpath('/imaging/astle/users/nz01/RED_MEG/Outputs_FullDataset/HMM_Outputs3/'); % data directory
addpath('/imaging/astle/users/nz01/RED_MEG/Scripts/MEG-ROI-nets/'); % ROI nets

hmm = load('hmm_k7_clean3.mat');
    hmm = hmm.hmm;
Gamma = load('Gamma_k7_clean3.mat');
    Gamma = Gamma.Gamma;
    
%% find all files matching data2_*.mat in the data directory

datdir='/imaging/astle/users/nz01/RED_MEG/Outputs_FullDataset/Preproc_Outputs3/';
datfiles=dir([datdir,'data2__*.mat']);
data={};
T={};

for f=1:numel(datfiles)
    
    disp(['Loading ' datfiles(f).name])
    
    % find the corresponding T_*.mat file
    Tfile=split(datfiles(f).name,'__');
    Tfile=['T__' Tfile{2}];
    
    if exist([datdir Tfile],'file')~=2
        % if a corresponding T__*.mat file doesn't exist, skip this iteration
        disp(['Cannot find corresponding T file, ' Tfile '. Skipping this participant'])
        continue        
    else
        d=load([datdir datfiles(f).name]);
        data{f}= d.a2;
        d=load([datdir Tfile]);
        T{f}=d.T;
    end  
    
end

data([4 9 29 39 41]) = []; % Exclude weird data here from participants 99013,99078,99112, and 99114
data(4) = []; % ADDITIONAL

Hz = 250;
no_states = 8;
T = load('T_51.mat');
    T = T.Tc.T;
    T([4 9 29 40]) = [];
    T(4) = []; % ADDITIONAL
data_modality = 'M/EEG power';

%% Spectral estimation - don't run until needed, it will take ages

if isempty(gcp('nocreate'))
    numworkers=24;
    P=cbupool(numworkers);
    %customise the pool options - increase the amount of memory per worker,
    %and increase the default time limit
    P.SubmitArguments=['--ntasks=' num2str(numworkers) ' --mem-per-cpu=20G --time=24:00:00'];
    parpool(P,numworkers);
end

options_spectra = struct(); 
options_spectra.Fs = Hz; % Sampling rate 
options_spectra.fpass = [1 30];  % band of frequency you're interested in
options_spectra.p = 0.1; % interval of confidence  
options_spectra.to_do = [1 0]; % turn off pdc
if strcmp(data_modality,'LFP') % MAR spectra
    options_spectra.to_do = [1 1];
    options_spectra.order = 15; 
    options_spectra.Nf = 90; 
    spectra = hmmspectramar(data,T,[],Gamma,options_spectra);
else % Multi-taper spectra
    options_spectra.tapers = [4 7]; % internal multitaper parameter
    options_spectra.win = 10; % useful to diagnose if the HMM is capturing dynamics or grand between-subject differences (see Wiki)
    options_spectra.embeddedlags = -14:14; % 30 seconds
    spectra = hmmspectramt(data,T,Gamma,options_spectra);
end

cd('/imaging/astle/users/nz01/RED_MEG/Outputs_FullDataset/HMM_Outputs3/K7');
save('spectra_k7_3.mat','spectra','-v7.3');

delete(gcp('nocreate')); % reset parallel pool

%% Frequency decompositions

if isempty(gcp('nocreate'))
    numworkers=24;
    P=cbupool(numworkers);
    %customise the pool options - increase the amount of memory per worker,
    %and increase the default time limit
    P.SubmitArguments=['--ntasks=' num2str(numworkers) ' --mem-per-cpu=20G --time=24:00:00'];
    parpool(P,numworkers);
end

% Build an HMM time-based frequency representation
[psd_tf,coh_tf,pdc_tf] = hmmtimefreq(spectra,Gamma,centered);

% Establish frequency bands
bands = [0 4; 4 8; 8 13; 13 Inf]; % Delta, Theta, Alpha, and Beta

% Do frequency decomposition into frequency models
sp_fit = spectbands(spectra,bands);

% Data-driven decomposition into frequency modes
options = hmm.train;
nsubjects = 46;
Gamma = padGamma(Gamma,T,options);
sp_fit_subj = cell(nsubjects,1); 
Tacc = 0;
for j = 1:nsubjects
  ind = Tacc + (1:T{j}); Tacc = Tacc + length(ind);
  sp_fit_subj{j} = hmmspectramt(X{j},T{j},hmm,Gamma(ind,:),options_spectra);
end

params_fac = struct();
params_fac.Method = 'NNMF';
params_fac.Ncomp = 4;
[spectral_factors,spectral_profiles,spectral_factors_subj] = spectdecompose(sp_fit_subj,params_fac);

save('spectralfactors_k7.mat','spectral_factors','-v7.3');
save('spectralprofiles_k7.mat','spectral_profiles','-v7.3');
save('spectralfactors_sub_k7.mat','spectral_factors_subj','-v7.3');

delete(gcp('nocreate')); % reset parallel pool