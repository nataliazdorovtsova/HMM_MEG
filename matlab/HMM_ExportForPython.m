%% Consolidated export of HMM summary outputs for the Python analysis package
%
% Replaces the scattered xlsread/csvread/csvwrite calls previously spread
% across HMM_CognitiveBehaviouralAnalyses.m, HMM_BetweenStatesAnalyses.m,
% and HMM_Entropy.m with one documented export step. This script assumes
% those three scripts (or their upstream computations) have already been
% run and that the summary variables below exist on the MATLAB path -
% it does not recompute FO/switching rate/entropy/etc. itself.
%
% Produces three tidy files consumed by the Python 'analysis' package:
%   subject_level.csv       - one row per subject
%   state_level.csv         - long format, one row per subject x state
%   transition_matrices.csv - long format, one row per subject x from_state x to_state

addpath('/imaging/astle/users/nz01/RED_MEG/Outputs_FullDataset/HMM_Outputs3/K7/'); % data directory

% Where the three CSVs should be written. Deliberately NOT `pwd`: MATLAB's
% `run('/path/to/this_script.m')` changes the current folder to this
% script's own folder before executing it (and back again afterwards), so
% `pwd` here silently resolves to wherever this .m file lives - which, if
% that happens to be a git-tracked repo, risks writing per-participant data
% into a public repository. Point this at a private, non-git location.
outputdir = '/imaging/astle/natalia/HMM_MEG_exports';

%% Load behavioural/cognitive data

data = xlsread('CALMRED_Data.xlsx');
data(4,:) = []; % exclude participant with MEG artifacts

nsub = size(data,1);
subject_id  = (1:nsub)';
age         = data(:,2);
sex         = data(:,3); % 1 = male, 2 = female
WASI        = data(:,5);
SDQ_total       = data(:,6);
SDQ_hyperactivity = data(:,7);
SDQ_conduct       = data(:,8);
SDQ_peerproblems  = data(:,9);
SDQ_emotion       = data(:,10);
SDQ_prosocial     = data(:,11);

%% Load HMM summary metrics (already computed by upstream scripts)

SwitchingRate = csvread('SwitchingRate.csv');
maxFO         = csvread('maxFO.csv');
entropy       = csvread('EntRate.csv');
FO            = csvread('FO.csv'); % nsub x nstates

nstates = size(FO,2);

T = csvread('T_46.csv'); % number of timepoints per subject, used to segment Xi/vpath below

Xi = load('Xi_k7_clean3.mat');
    Xi = Xi.Xi;

vpath = load('viterbipath_k7_clean3.mat');
    vpath = vpath.vpath; % Viterbi state path, one entry per timepoint, concatenated across subjects

%% subject_level.csv

subject_level = table(subject_id, age, sex, WASI, SDQ_total, SDQ_hyperactivity, ...
    SDQ_conduct, SDQ_peerproblems, SDQ_emotion, SDQ_prosocial, ...
    SwitchingRate, entropy, maxFO, ...
    'VariableNames', {'subject_id','age','sex','WASI_T','SDQ_total','SDQ_hyperactivity', ...
    'SDQ_conduct','SDQ_peerproblems','SDQ_emotion','SDQ_prosocial', ...
    'switching_rate','entropy_rate','max_FO'});

writetable(subject_level, fullfile(outputdir,'subject_level.csv'));

%% state_level.csv (long format: subject_id, state, FO, lifetime_mean, interval_mean)

state_level = table();
row = 1;
for s = 1:nstates
    for i = 1:nsub
        state_level.subject_id(row) = subject_id(i);
        state_level.state(row) = s;
        state_level.FO(row) = FO(i,s);
        row = row + 1;
    end
end

% Per-subject, per-state mean lifetime/interval, derived fresh from the
% Viterbi path rather than from Intervals_k7_3.mat/LifeTimes_k7_3.mat -
% those only ever stored per-state vectors concatenated across all subjects
% with no subject index, so a genuine per-subject mean was not recoverable
% from them. vpath (segmented by T, same pattern as the Xi segmentation
% below) has everything needed: a "lifetime" is the length of one
% continuous run in a state; an "interval" is the gap between the end of
% one run and the start of the next run in that same state. No minimum-
% duration threshold is applied here (the original ~50ms exclusion was
% flagged by Reviewer #3 as unmotivated) - compare with/without one if
% needed once real numbers are in hand.
vpath_by_subject = cell(nsub,1);
time_passed = 1;
for x = 1:nsub
    vpath_by_subject{x} = vpath(time_passed : time_passed + T(x) - 1);
    time_passed = time_passed + T(x);
end

lifetime_mean = nan(nsub,nstates);
interval_mean = nan(nsub,nstates);

for x = 1:nsub
    path = vpath_by_subject{x};
    for s = 1:nstates
        in_state = (path == s);
        d = diff([0; in_state(:); 0]);
        run_starts = find(d == 1);
        run_ends = find(d == -1) - 1;

        if ~isempty(run_starts)
            lifetime_mean(x,s) = mean(run_ends - run_starts + 1); % in samples
        end
        if numel(run_starts) > 1
            interval_mean(x,s) = mean(run_starts(2:end) - run_ends(1:end-1) - 1); % in samples
        end
        % Both stay NaN if subject x never visits state s at all - a
        % genuine missing value (see Reviewer #3 concern 2c), not a
        % fabricated zero.
    end
end

state_level.lifetime_mean = reshape(lifetime_mean, [], 1);
state_level.interval_mean = reshape(interval_mean, [], 1);

writetable(state_level, fullfile(outputdir,'state_level.csv'));

%% transition_matrices.csv (long format: subject_id, from_state, to_state, probability)

Tnew = T - 1;
Xi2 = cell(nsub,1);
Time_passed = 1;
for x = 1:nsub
    Xi2{x} = Xi((Time_passed:Time_passed + Tnew(x) - 1),:,:);
    Time_passed = Time_passed + Tnew(x);
end

transition_rows = table();
row = 1;
for x = 1:nsub
    Ptmp = squeeze(sum(Xi2{x},1));
    for s = 1:nstates
        Ptmp(s,:) = Ptmp(s,:)/sum(Ptmp(s,:));
    end
    for from_state = 1:nstates
        for to_state = 1:nstates
            transition_rows.subject_id(row) = subject_id(x);
            transition_rows.from_state(row) = from_state;
            transition_rows.to_state(row) = to_state;
            transition_rows.probability(row) = Ptmp(from_state,to_state);
            row = row + 1;
        end
    end
end

writetable(transition_rows, fullfile(outputdir,'transition_matrices.csv'));

disp('Wrote subject_level.csv, state_level.csv, and transition_matrices.csv');
