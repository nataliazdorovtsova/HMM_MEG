% Cognitive and behavioural analyses 

addpath('/imaging/astle/users/nz01/RED_MEG/Outputs_FullDataset/HMM_Outputs3/K7/'); % data directory

%% Load data

data = xlsread('CALMRED_Data.xlsx');
data(4,:) = []; % exclude participant with MEG artifacts
    age = data(:,2);
    gender = data(:,3);
    WASI = data(:,5);
    SDQ = data(:,6);
    hyperactivity = data(:,7);
    conduct = data(:,8);
    peerproblems = data(:,9);
    emotion = data(:,10);
    prosocial = data(:,11);
SwitchingRate = csvread('SwitchingRate.csv');
FO = csvread('FO.csv');
maxFO = csvread('maxFO.csv');
entropy = csvread('EntRate.csv');

%% Get gender data
maleind = data(:,3) == 1;
femaleind = data(:,3) == 2;

femaledata = data(femaleind,:);
maledata = data(maleind,:);

%% Some plots

% Heatmap of correlations between cognitive and behavioural scores
CogBehaveCorr = corrcoef(data(:,5:11));
figure;
imagesc(CogBehaveCorr)
b = gca;
b.TickDir = 'out';
b.FontName = 'Arial';
b.FontSize = 12;
title('Correlations between cognitive and behavioural scores')
colormap('parula');
colorbar('southoutside');
set(gcf,'color','w');
box off;

% WASI histogram
figure;
histogram(WASI)
b = gca;
b.TickDir = 'out';
b.FontName = 'Arial';
b.FontSize = 12;
xlabel('WASI-II MR T-Score');
ylabel('Number of Participants');
set(gcf,'color','w');
box off;

% Age histogram
figure;
histogram(age)
b = gca;
b.TickDir = 'out';
b.FontName = 'Arial';
b.FontSize = 12;
xlabel('Years of Age');
ylabel('Number of Participants');
set(gcf,'color','w');
box off;

% Gender histogram
figure;
histogram(gender)
b = gca;
b.TickDir = 'out';
b.FontName = 'Arial';
b.FontSize = 12;
xlabel('Gender');
ylabel('Number of Participants');
set(gcf,'color','w');
box off;

% SDQ Total histogram
figure;
histogram(SDQ)
b = gca;
b.TickDir = 'out';
b.FontName = 'Arial';
b.FontSize = 12;
xlabel('SDQ (Total Score)');
ylabel('Number of Participants');
set(gcf,'color','w');
box off;

% SDQ Hyperactivity histogram
figure;
histogram(hyperactivity)
b = gca;
b.TickDir = 'out';
b.FontName = 'Arial';
b.FontSize = 12;
xlabel('SDQ Hyperativity Score');
ylabel('Number of Participants');
set(gcf,'color','w');
box off;

%% Gender analyses

% FO for each state
for i= 1:7
[H,P,CI,STATS]= ttest2(FO(femaleind,i),FO(maleind,i));
Pvals(i)=P;
Tvals(i)=STATS.tstat;
end
[padj,alpha] = multicmp(Pvals,'fdr',0.05)

% max FO
[H,P,CI,STATS]= ttest2(maxFO(femaleind,:),maxFO(maleind,:))

% switching rate 
[H,P,CI,STATS]= ttest2(SwitchingRate(femaleind,:),SwitchingRate(maleind,:))

% Entropy rate
[H,P,CI,STATS]= ttest2(entropy(femaleind,:),entropy(maleind,:))


%% Age analyses

% FO for each state
clear stats.p
clear stats.t
for i= 1:7
[b, dev, stats]=glmfit([age,gender],FO(:,i));
Pvals(i)=stats.p(2);
Tvals(i)=stats.t(2);
end

[padj,alpha] = multicmp(Pvals,'fdr',0.05)

% max FO
clear stats.p
clear stats.t
[b, dev, stats]=glmfit([age,gender],maxFO);
Pvals=stats.p(2);
Tvals=stats.t(2);
[padj,alpha] = multicmp(Pvals,'fdr',0.05)

% switching rate 
clear stats.p
clear stats.t
[b, dev, stats]=glmfit([age,gender],SwitchingRate);
Pvals=stats.p(2);
Tvals=stats.t(2);
[padj,alpha] = multicmp(Pvals,'fdr',0.05)

% Entropy rate
clear stats.p
clear stats.t
[b, dev, stats]=glmfit([age,gender],entropy);
Pvals=stats.p(2);
Tvals=stats.t(2);
[padj,alpha] = multicmp(Pvals,'fdr',0.05)

%% Cognitive analyses

% FO for each state
clear stats.p
clear stats.t
for i= 1:7
[b, dev, stats]=glmfit([WASI,age,gender],FO(:,i));
Pvals(i)=stats.p(2);
Tvals(i)=stats.t(2);
end
[padj,alpha] = multicmp(Pvals,'fdr',0.05)
[r p] = corrcoef(WASI,FO(:,1))
[r p] = corrcoef(WASI,FO(:,3))
[r p] = corrcoef(WASI,FO(:,4))
[r p] = corrcoef(WASI,FO(:,6))
[r p] = corrcoef(WASI,FO(:,7))

% max FO
clear stats.p
clear stats.t
[b, dev, stats]=glmfit([WASI,age,gender],maxFO);
Pvals=stats.p(2);
Tvals=stats.t(2);
[padj,alpha] = multicmp(Pvals,'fdr',0.05)
[r p] = corrcoef(WASI,maxFO)

% switching rate 
clear stats.p
clear stats.t
[b, dev, stats]=glmfit([WASI,age,gender],SwitchingRate);
Pvals=stats.p(2);
Tvals=stats.t(2);
[padj,alpha] = multicmp(Pvals,'fdr',0.05)
[r, p] = corrcoef(WASI, SwitchingRate)

% Entropy rate
clear stats.p
clear stats.t
[b, dev, stats]=glmfit([WASI,age,gender],entropy);
Pvals=stats.p(2);
Tvals=stats.t(2);
[padj,alpha] = multicmp(Pvals,'fdr',0.05)
[r, p] = corrcoef(WASI,entropy)

% Correlation between entropy and switching rate (r = 0.9979, p < 0.00001)
[r,p] = corrcoef(SwitchingRate,entropy)

%% Behavioural analyses

% FO for each state
clear stats.p
clear stats.t
for i= 1:7
[b, dev, stats]=glmfit([prosocial,age,gender],FO(:,i));
Pvals(i)=stats.p(2);
Tvals(i)=stats.t(2);
end
[padj,alpha] = multicmp(Pvals,'fdr',0.05)

% max FO
clear stats.p
clear stats.t
[b, dev, stats]=glmfit([prosocial,age,gender],maxFO);
Pvals=stats.p(2);
Tvals=stats.t(2);
[padj,alpha] = multicmp(Pvals,'fdr',0.05)

% switching rate 
clear stats.p
clear stats.t
[b, dev, stats]=glmfit([prosocial,age,gender],SwitchingRate);
Pvals=stats.p(2);
Tvals=stats.t(2);
[padj,alpha] = multicmp(Pvals,'fdr',0.05)
disp(Tvals)

% Entropy rate
clear stats.p
clear stats.t
[b, dev, stats]=glmfit([prosocial,age,gender],entropy);
Pvals=stats.p(2);
Tvals=stats.t(2);
[padj,alpha] = multicmp(Pvals,'fdr',0.05)
disp(Tvals)

%% WASI and switching rate
model1 = fitlm(WASI,SwitchingRate);
figure;
p = plot(model1);
p(1).Marker = 'o';
p(1).MarkerEdgeColor = 'cyan';
p(1).MarkerFaceColor = 'blue';
p(1).MarkerSize = 7;
p(2).LineWidth = 3;
p(3).LineWidth = 2;
p(4).LineWidth = 2;
title('Switching Rates as a function of WASI scores')
xlabel('WASI-II T-Score')
ylabel('Switching Rate')
ylim([0 0.1])
b = gca;
b.TickDir = 'out';
b.FontName = 'Arial';
b.FontSize = 14;
set(gcf,'color','w');
set(gcf,'Position',[100 100 800 600]);
box off;

%% WASI and Entropy
model2 = fitlm(WASI,entropy);
figure;
p = plot(model2);
p(1).Marker = 'o';
p(1).MarkerEdgeColor = 'cyan';
p(1).MarkerFaceColor = 'blue';
p(1).MarkerSize = 7;
p(2).LineWidth = 3;
p(3).LineWidth = 2;
p(4).LineWidth = 2;
title('Entropy Rates as a function of WASI scores')
xlabel('WASI-II T-Score')
ylabel('Entropy Rate (Nats/Second)')
ylim([0 2])
b = gca;
b.TickDir = 'out';
b.FontName = 'Arial';
b.FontSize = 14;
set(gcf,'color','w');
set(gcf,'Position',[100 100 800 600]);
box off;

%% Switching Rate and Entropy
model3 = fitlm(SwitchingRate,entropy);
figure;
p = plot(model3);
p(1).Marker = 'o';
p(1).MarkerEdgeColor = 'cyan';
p(1).MarkerFaceColor = 'blue';
p(1).MarkerSize = 7;
p(2).LineWidth = 3;
p(3).LineWidth = 2;
p(4).LineWidth = 2;
title('Entropy Rates as a function of Switching Rates')
xlabel('Switching Rate')
ylabel('Entropy Rate')
ylim([0 2])
b = gca;
b.TickDir = 'out';
b.FontName = 'Arial';
b.FontSize = 14;
set(gcf,'color','w');
set(gcf,'Position',[100 100 800 600]);
box off;

%% WASI and Maximum FO
model4 = fitlm(WASI,maxFO);
figure;
p = plot(model4);
p(1).Marker = 'o';
p(1).MarkerEdgeColor = 'cyan';
p(1).MarkerFaceColor = 'blue';
p(1).MarkerSize = 7;
p(2).LineWidth = 3;
p(3).LineWidth = 2;
p(4).LineWidth = 2;
title('Maximum Fractional Occupancies as a function of WASI scores')
xlabel('WASI-II T-Score')
ylabel('Maximum Fractional Occupancy')
ylim([0 1])
b = gca;
b.TickDir = 'out';
b.FontName = 'Arial';
b.FontSize = 14;
set(gcf,'color','w');
set(gcf,'Position',[100 100 800 600]);
box off;

%% WASI and FO (state 1)
model5 = fitlm(WASI,FO(:,1));
figure;
p = plot(model5);
p(1).Marker = 'o';
p(1).MarkerEdgeColor = 'cyan';
p(1).MarkerFaceColor = 'blue';
p(1).MarkerSize = 7;
p(2).LineWidth = 3;
p(3).LineWidth = 2;
p(4).LineWidth = 2;
title('Fractional Occupancies (State 1) as a function of WASI scores')
xlabel('WASI-II T-Score')
ylabel('Fractional Occupancy')
ylim([0 0.1])
b = gca;
b.TickDir = 'out';
b.FontName = 'Arial';
b.FontSize = 14;
set(gcf,'color','w');
set(gcf,'Position',[100 100 800 600]);
box off;

%% WASI and FO (state 3)
model6 = fitlm(WASI,FO(:,3));
figure;
p = plot(model6);
p(1).Marker = 'o';
p(1).MarkerEdgeColor = 'cyan';
p(1).MarkerFaceColor = 'blue';
p(1).MarkerSize = 7;
p(2).LineWidth = 3;
p(3).LineWidth = 2;
p(4).LineWidth = 2;
title('Fractional Occupancies (State 3) as a function of WASI scores')
xlabel('WASI-II T-Score')
ylabel('Fractional Occupancy')
ylim([0 0.8])
b = gca;
b.TickDir = 'out';
b.FontName = 'Arial';
b.FontSize = 14;
set(gcf,'color','w');
set(gcf,'Position',[100 100 800 600]);
box off;

%% WASI and FO (state 4)
model7 = fitlm(WASI,FO(:,4));
figure;
p = plot(model7);
p(1).Marker = 'o';
p(1).MarkerEdgeColor = 'cyan';
p(1).MarkerFaceColor = 'blue';
p(1).MarkerSize = 7;
p(2).LineWidth = 3;
p(3).LineWidth = 2;
p(4).LineWidth = 2;
title('Fractional Occupancies (State 4) as a function of WASI scores')
xlabel('WASI-II T-Score')
ylabel('Fractional Occupancy')
ylim([0 0.5])
b = gca;
b.TickDir = 'out';
b.FontName = 'Arial';
b.FontSize = 14;
set(gcf,'color','w');
set(gcf,'Position',[100 100 800 600]);
box off;

%% WASI and FO (state 6)
model8 = fitlm(WASI,FO(:,6));
figure;
p = plot(model8);
p(1).Marker = 'o';
p(1).MarkerEdgeColor = 'cyan';
p(1).MarkerFaceColor = 'blue';
p(1).MarkerSize = 7;
p(2).LineWidth = 3;
p(3).LineWidth = 2;
p(4).LineWidth = 2;
title('Fractional Occupancies (State 6) as a function of WASI scores')
xlabel('WASI-II T-Score')
ylabel('Fractional Occupancy')
ylim([0 0.5])
b = gca;
b.TickDir = 'out';
b.FontName = 'Arial';
b.FontSize = 14;
set(gcf,'color','w');
set(gcf,'Position',[100 100 800 600]);
box off;

%% WASI and FO (state 7)
model9 = fitlm(WASI,FO(:,7));
figure;
p = plot(model9);
p(1).Marker = 'o';
p(1).MarkerEdgeColor = 'cyan';
p(1).MarkerFaceColor = 'blue';
p(1).MarkerSize = 7;
p(2).LineWidth = 3;
p(3).LineWidth = 2;
p(4).LineWidth = 2;
title('Fractional Occupancies (State 7) as a function of WASI scores')
xlabel('WASI-II T-Score')
ylabel('Fractional Occupancy')
ylim([0 0.5])
b = gca;
b.TickDir = 'out';
b.FontName = 'Arial';
b.FontSize = 14;
set(gcf,'color','w');
set(gcf,'Position',[100 100 800 600]);
box off;