%% HMM Entropy rate calculations for each participant

Xi = load('Xi_k7_clean3.mat');
    Xi = Xi.Xi;
    
%% First, we load in the timesteps so that we know how to segment Xi

T = csvread('T_46.csv');

%% To compare these timepoints with Xi, we will need to subtract 1 from each of the columns 
% (# of timepoints for each participant) because Xi holds posterior
%  probabilities for state pairs (so we end up with n-1 for each participant)

for i = 1:46
    
    Tnew(:,i) = T(i)-1;
    
end

%% Now, we need to segment Xi per participant

Xi2 = cell(46,1);
Time_passed = 1;

for x = 1:46
    
    Xi2{x} = Xi((Time_passed:Time_passed + Tnew(x) - 1),:,:);
    Time_passed = Time_passed + Tnew(x);
    
end

% Xi2 is now a cell array with values for each of the 46 participants, which are
% organised as timepoints x nstates x nstates. These are posterior joint
% probabilities of pairs of states across each participant's timecourse

%% Now we need to calculate the transition matrix for each participant

P = zeros(46,7,7); % array containing transition probability matrices for each participant
% P(id, state1, state2) is the probability that participant 'id'
% transitions from state1 to state2.
% In particular, the sum over s of P(id, state1, s) should be 1. 

% Loop through participants, finding the transition matrix for each one.
for x = 1:46
    P(x,:,:) = sum(Xi2{x},1);
    % Normalise
    for s = 1:7
        P(x,s,:) = P(x,s,:)/sum(P(x,s,:));
    end
    
end

%% From the transition matrix, compute the stationary distribution (fractional occupancy)

mu = zeros(46,7);

for x = 1:46
    P_temp = reshape(P(x,:,:),[7,7]);
    mu(x,:) = null(P_temp' - eye(7));
    mu(x,:) = mu(x,:)/sum(mu(x,:));
end

%% Next, we calculate the entropy rate

Ent_rate = zeros(46,1);
Ent_state = zeros(7,1);

for x = 1:46
    P_temp = reshape(P(x,:,:),[7,7]);
    
    % Calculate the entropy for each state
    for s = 1:7
        Ent_state(s) = -dot(P_temp(s,:),log(P_temp(s,:)));
    end
    
    % Average over the stationary distribition 
    Ent_rate(x) = dot(mu(x,:),Ent_state); % Nats per unit time - multiply by 4 to get Nats/second
    
end

Ent_rate_secs = Ent_rate*4; % converts to Nats/second

csvwrite('EntRate.csv',Ent_rate_secs);

%% Make a histogram for the entropy rates

figure;
histogram(Ent_rate_secs)
b = gca;
b.TickDir = 'out';
b.FontName = 'Arial';
b.FontSize = 12;
set(gcf,'color','w');
xlabel('Entropy Rate (Nats/Second)');
ylabel('Number of Participants');
box off;

%% plot some graphs

% set number of states
nstate = 7;
% set the sub
sub = 2; % if single subject below
% set the plot
% A = normalize(squeeze(P(sub,:,:)),'range'); % single subject
A = normalize(squeeze(mean(P,1)),'range'); % mean
g = digraph(A);
% state labels
state_labels = {'State 1 (DMN+)','State 2(VT+)',...
                'State 3 (DMN-)','State 4 (SM+FP-)',...
                'State 5 (V+FT-)','State 6 (FP+V-)','State 7 (LP+DMN-)'};

%%  Plot state transitions (leaving)

Ao = A;
Ao(find(eye(nstate))) = 0;
x = 1; % keep top x %
k = sort(Ao(:),'descend');
i = k(round(x*(nstate^2)));
g = digraph(Ao.*(Ao>i));

% set colours
u = 1+round(63*normalize(g.Edges.Weight,'range'));
h = colormap('Parula'); % edge colour
h = h(u,:);

figure;
plot(g,...
    'LineWidth',70*g.Edges.Weight,...
    'MarkerSize',50*sum(Ao),...
    'NodeLabel',state_labels,...
    'MarkerSize',150*mean(FO),...
    'ArrowSize',14,...
    'EdgeColor',h,...
    'NodeColor',[0.9290 0.6940 0.1250],...
    'EdgeAlpha',1);
%colorbar('southoutside');
set(gcf,'color','w');
axis off;

%% Calculate correlation between each transition and cognition

cognition_r = [];
pvals = [];

for i = 1:nstate
    for j = 1:nstate
        t = squeeze(P(:,i,j));
        [cognition_r(i,j) pvals(i,j)] = corr(t,WASI);
    end
end

% make colours
data2plot = cognition_r .* (pvals<0.025);
dg = digraph(data2plot);
% set colours
u = 1+round(63*normalize(dg.Edges.Weight,'range'));
newmap = brewermap([],'PiYG');
h = newmap(u,:);

figure;
imagesc(cognition_r)
b = gca;
b.TickDir = 'out';
b.FontName = 'Arial';
b.FontSize = 12;
title('Cognitive ability and state transition probabilities')
colormap(newmap);
colorbar('southoutside');
set(gcf,'color','w');
box off;

%% Plot transition probability & cognition correlations as colours on graph

figure; 
y = plot(dg,...
    'LineWidth',5,...
    'EdgeColor',h,...
    'NodeColor',[0.9290 0.6940 0.1250],...
    'ArrowSize',20,...
    'NodeLabel',state_labels,...
    'MarkerSize',150*mean(FO));
set(gcf,'color','w');
axis off; b = gca; b.FontName = 'Arial'; b.FontSize = 14;

%% Column-wise comparisons between cognition and transitioning into a state

cognition_state = [];
pvals2 = [];
for i = 1:nstate
    t = squeeze(sum(P(:,:,i),2)); % transition to a particular state
    [cognition_state(i) pvals2(i)] = corr(t,WASI)
end
[padj,alpha] = multicmp(pvals2,'fdr',0.05)

% transitions to states 3, 4,and 6 are significantly associated with cognitive ability

% State 3 - negative relationship
state3prob = fitlm(WASI,squeeze(sum(P(:,:,3),2)));
figure;
p = plot(state3prob);
p(1).Marker = 'o';
p(1).MarkerEdgeColor = 'cyan';
p(1).MarkerFaceColor = 'blue';
p(1).MarkerSize = 7;
p(2).LineWidth = 3;
p(3).LineWidth = 2;
p(4).LineWidth = 2;
title('Cognition and Transitions Into State 3 (DMN-)')
xlabel('WASI-II T-Score')
ylabel('Total sum of transition probabilities to state')
ylim([0.9 1.2])
b = gca;
b.TickDir = 'out';
b.FontName = 'Arial';
b.FontSize = 14;
set(gcf,'color','w');
set(gcf,'Position',[100 100 800 600]);
box off;

% State 4 - positive relationship
state4prob = fitlm(WASI,squeeze(sum(P(:,:,4),2)));
figure;
p = plot(state4prob);
p(1).Marker = 'o';
p(1).MarkerEdgeColor = 'cyan';
p(1).MarkerFaceColor = 'blue';
p(1).MarkerSize = 7;
p(2).LineWidth = 3;
p(3).LineWidth = 2;
p(4).LineWidth = 2;
title('Cognition and Transitions Into State 4 (SM+FP-)')
xlabel('WASI-II T-Score')
ylabel('Total sum of transition probabilities to state')
ylim([0.9 1.2])
b = gca;
b.TickDir = 'out';
b.FontName = 'Arial';
b.FontSize = 14;
set(gcf,'color','w');
set(gcf,'Position',[100 100 800 600]);
box off;

% State 6 - positive relationship
state6prob = fitlm(WASI,squeeze(sum(P(:,:,6),2)));
figure;
p = plot(state6prob);
p(1).Marker = 'o';
p(1).MarkerEdgeColor = 'cyan';
p(1).MarkerFaceColor = 'blue';
p(1).MarkerSize = 7;
p(2).LineWidth = 3;
p(3).LineWidth = 2;
p(4).LineWidth = 2;
title('Cognition and Transitions Into State 6 (FP+V-)')
xlabel('WASI-II T-Score')
ylabel('Total sum of transition probabilities to state')
ylim([0.9 1.2])
b = gca;
b.TickDir = 'out';
b.FontName = 'Arial';
b.FontSize = 14;
set(gcf,'color','w');
set(gcf,'Position',[100 100 800 600]);
box off;

%% Correlate modularity (Q) of transition matrices with communicability

% loop over subjects
% for each subject compute the communicability over the top x % transitions
nsub = 46;
% initalise
communicability = [];
communities = [];
quality = [];

for sub = 1:nsub
    % get their matrix
    Ao = normalize(squeeze(P(sub,:,:)),'range');
    Ao(find(eye(nstate))) = 0;
    x = 1; % keep top x %
    k = sort(Ao(:),'descend');
    i = k(round(x*(nstate^2)));
    A = Ao.*(Ao>i);
    % compute the communicability
    communicability(sub) = sum(sum(expm(A),'omitnan'));
    % compute the modularity
    [communities(sub,:),quality(sub)] = modularity_dir(A);
end

[r p] = corr(communicability',quality')

% Plot correlation
model10 = fitlm(communicability',quality');
figure;
p = plot(model10);
p(1).Marker = 'o';
p(1).MarkerEdgeColor = 'cyan';
p(1).MarkerFaceColor = 'blue';
p(1).MarkerSize = 7;
p(2).LineWidth = 3;
p(3).LineWidth = 2;
p(4).LineWidth = 2;
title('Communicability and Transition Matrix Modularity')
xlabel('Communicability')
ylabel('Modularity (Q)')
ylim([0 0.5])
b = gca;
b.TickDir = 'out';
b.FontName = 'Arial';
b.FontSize = 14;
set(gcf,'color','w');
set(gcf,'Position',[100 100 800 600]);
box off;


%% Calculate relationship between communicability & quality and cognition

[r1 p1] = corr(communicability',WASI)
[r2 p2] = corr(quality',WASI)
