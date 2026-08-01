%% Set environment and create paths

setenv('FSLDIR','/imaging/local/software/fsl/v5.0.7/x86_64/fsl/'); % FSL
addpath(genpath('/imaging/astle/users/nz01/RED_MEG/Scripts/HMM-MAR-master/')); % scripts
addpath(genpath('/imaging/astle/users/nz01/RED_MEG/Scripts/SPM_Files')); % data directory
addpath(genpath('/imaging/local/software/spm_cbu_svn/releases/spm12_latest/')); % SPM v12

%% Set up parpool - hmmmar should find this using the gcp function
if isempty(gcp('nocreate'))
    numworkers=24;
    P=cbupool(numworkers);
    %customise the pool options - increase the amount of memory per worker,
    %and increase the default time limit
    P.SubmitArguments=['--ntasks=' num2str(numworkers) ' --mem-per-cpu=20G --time=24:00:00'];
    parpool(P,numworkers);
end

%% Extract data from structs and create new files

%datdir='/imaging/astle/users/nz01/RED_MEG/Outputs_FullDataset/Preproc_Outputs3/';
%datold = dir([datdir,'data__*.mat']);

%for g = 1:numel(datold)

%    label = labels{1,g}; % get labels from old HMM matlab script
%    disp(['Loading ' datold(g).name])
%    d=load([datdir datold(g).name]);
%    a2 = d.data.X; % new, indexed data
%    disp(['Saving new version of ' datold(g).name])
%    save(sprintf('data2_%s.mat',label),'a2','-v7.3'); % data output
    
%end

%% Cut out bad segments

% participant 99013
% a2 = load('data2__99013.mat');
%    a2 = a2.a2;
%    a2(1:14000,:) = [];
%    save('data2__99013.mat','a2');

% participant 99020
% a2 = load('data2__99020.mat');
%    a2 = a2.a2;
%    a2(1:13500,:) = [];
%    save('data2__99020.mat','a2');
    
% participant 99078
% a2 = load('data2__99078.mat');
%    a2 = a2.a2;
%    a2(1:18500,:) = [];
%    save('data2__99078.mat','a2');   

% participant 99119
% a2 = load('data2__99119.mat');
%    a2 = a2.a2;
%    a2(1:11000,:) = [];
%    save('data2__99119.mat','a2');  
    
% participant calm02
%a2 = load('data2__calm02.mat');
%    a2 = a2.a2;
%    a2(87501:102500,:) = [];
%    save('data2__calm02.mat','a2'); 
    
% participant calm03
%a2 = load('data2__calm03.mat');
%    a2 = a2.a2;
%    a2(1:37500,:) = [];
%    save('data2__calm03.mat','a2');
    
% participant calm04
%a2 = load('data2__calm04.mat');
%    a2 = a2.a2;
%    a2(1:20000,:) = [];
%    a2(end-4999:end,:) = [];
%    save('data2__calm04.mat','a2');     
    

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

%% Cut time segments for participants with bad data

T(4) = {142500}; % 99013
T(9) = {151000}; % 99020
T(29) = {137000}; % 99078
T(41) = {144250}; % 99119

T(48) = {140500}; % calmred02
T(49) = {118500}; % calmred03
T(50) = {133750}; % calmred04

data([4 9 29 39 41]) = []; % Exclude weird data here from participants 99013, 99029, 99078, 99112 (coregistration bad), and 99114
T([4 9 29 39 41]) = []; % All the participants with weird maxFO (all except for 39) had a lot of segments excluded - interesting...

% Exclude additional participant whose max FO was too high, indicating corrupded data
data(4) = [];
T(4) = [];
%% check data structure etc.

outputdir = '/imaging/astle/users/nz01/RED_MEG/Outputs_FullDataset/HMM_Outputs3/K6'; % where outputs should go

if ~exist('data','var') || ~exist('T','var')
    error('You need to load the data (data and T - see Documentation)')
end

data_modality = 'M/EEG power' ; % one of: 'fMRI', 'M/EEG', or 'M/EEG power' 
no_states = 6; % the number of states depends a lot on the question at hand
Hz = 250; % the frequency of the data
stochastic_inference = 0; % set to 1 if a normal run is too computationally expensive (memory or time)
N = length(T); % number of subjects

% getting the number of channels
if iscellstr(data) 
    dfilenames = data;
    if ~isempty(strfind(dfilenames{1},'.mat')), load(dfilenames{1},'X');
    else X = dlmread(dfilenames{1});
    end
elseif iscell(data)
    X = data{1};
end
ndim = size(X,2); 

%% Setting the options

options = struct();
options.K = no_states;
options.verbose = 1;
options.Fs = Hz;

if iscell(T), sumT = 0; for j = 1:N, sumT = sumT + sum(T{j}); end
else, sumT = sum(T); 
end

if strcmp(data_modality,'fMRI') % Gaussian observation model
    options.order = 0;
    options.zeromean = 0;
    options.covtype = 'full';     

elseif strcmp(data_modality,'M/EEG power') % Gaussian observation model**********
    options.order = 0;
    options.zeromean = 0;
    options.covtype = 'full';     
    options.onpower = 1; 
    
elseif strcmp(data_modality,'M/EEG') && ndim > 10 % Embedded observation model
    options.order = 0;
    options.zeromean = 0;
    options.covtype = 'full';
    options.embeddedlags = -14:14; % length of the state - 30ms

elseif strcmp(data_modality,'M/EEG') && ndim <= 10 % MAR observation model
    options.order = 5;
    options.zeromean = 1;
    options.covtype = 'diag';
    
else
    error('Option data_modality not recognised')

end

% stochastic options
if stochastic_inference
    options.BIGNbatch = 5;
    options.BIGtol = 1e-5;
    options.BIGcyc = 1000;
    options.BIGmincyc = 10;
    options.BIGundertol_tostop = 5;
    options.BIGforgetrate = 0.7;
    options.BIGbase_weights = 0.9;
    options.BIGverbose = 1;
end

%% HMM computation
[hmm, Gamma, Xi, vpath, fehist] = hmmmar(data,T,options);

% hmm: a structure with the estimated HMM-MAR model.

% Gamma: a (no. of time points X no. of states) matrix containing the state time courses. 

% Xi: a (no. of time points X no. of states X no. of states) matrix joint probability of past 
% and future states conditioned on data. Xi has one row less per trial than Gamma.

% vpath: a (no. of time points X 1) vector with the Viterbi path.

% GammaInit: the state time courses used after initialisation.

% residuals: if the model is trained on the residuals (which happens if some of the connections 
%have been clamped to a global value so that they do not drive the states), the value of such residuals.

% fehist: historial of the free energy, with one element per iteration of the variational inference.

%%  saving directory
 cd(outputdir);
 save('hmm_k6_clean3.mat','hmm','-v7.3'); 
 save('Gamma_k6_clean3.mat','Gamma','-v7.3'); 
 save('Xi_k6_clean3.mat','Xi','-v7.3'); 
 save('viterbipath_k6_clean3.mat','vpath','-v7.3');
 %save('GammaInit.mat','GammaInit','-v7.3'); 
 %save('residuals.mat','residuals','-v7.3'); 
 save('fehist_k6_clean3.mat','fehist','-v7.3');
 
 delete(gcp('nocreate')); % reset parallel pool
 