%% Between-states analyses - load data

intervals = load('Intervals_k7_3.mat');
    intervals = intervals.Intervals;
lifetimes = load('LifeTimes_k7_3.mat');
    lifetimes = lifetimes.LifeTimes;
FO = load('FO_k7_3.mat');
    FO = FO.FO;
    
I1 = intervals{1,1};
I2 = intervals{1,2};
I3 = intervals{1,3};
I4 = intervals{1,4};
I5 = intervals{1,5};
I6 = intervals{1,6};
I7 = intervals{1,7};
    
LT1 = lifetimes{1,1};
LT2 = lifetimes{1,2};
LT3 = lifetimes{1,3};
LT4 = lifetimes{1,4};
LT5 = lifetimes{1,5};
LT6 = lifetimes{1,6};
LT7 = lifetimes{1,7};

%% Statistical tests

% FOs - ANOVA
[p,tbl,stats] = anova1(FO);


%% create csv files to generate plots using Python Seaborn    

% Intervals  

Intervals = cat(2,I1,I2,I3,I4,I5,I6,I7); 
Intervals = transpose(Intervals);
IntervalsFull = zeros(length(Intervals),2);

for i = 1:length(Intervals)
    IntervalsFull(i,2) = Intervals(i);
end

IntervalsFull(1:length(I1),1) = 1;
IntervalsFull(1+length(I1):length(I1)+length(I2),1) = 2;
IntervalsFull(1+length(I1)+length(I2):length(I1)+length(I2)+length(I3),1) = 3;
IntervalsFull(1+length(I1)+length(I2)+length(I3):length(I1)+length(I2)+length(I3)+length(I4),1) = 4;
IntervalsFull(1+length(I1)+length(I2)+length(I3)+length(I4):length(I1)+length(I2)+length(I3)+length(I4)+length(I5),1) = 5;
IntervalsFull(1+length(I1)+length(I2)+length(I3)+length(I4)+length(I5):length(I1)+length(I2)+length(I3)+length(I4)+length(I5)+length(I6),1) = 6;
IntervalsFull(1+length(I1)+length(I2)+length(I3)+length(I4)+length(I5)+length(I6):end,1) = 7;

csvwrite('IntervalsFull.csv',IntervalsFull);

% Lifetimes

Lifetimes = cat(2,LT1,LT2,LT3,LT4,LT5,LT6,LT7); 
Lifetimes = transpose(Lifetimes);
LifetimesFull = zeros(length(Lifetimes),2);

for i = 1:length(Lifetimes)
    LifetimesFull(i,2) = Lifetimes(i);
end

LifetimesFull(1:length(LT1),1) = 1;
LifetimesFull(1+length(LT1):length(LT1)+length(LT2),1) = 2;
LifetimesFull(1+length(LT1)+length(LT2):length(LT1)+length(LT2)+length(LT3),1) = 3;
LifetimesFull(1+length(LT1)+length(LT2)+length(LT3):length(LT1)+length(LT2)+length(LT3)+length(LT4),1) = 4;
LifetimesFull(1+length(LT1)+length(LT2)+length(LT3)+length(LT4):length(LT1)+length(LT2)+length(LT3)+length(LT4)+length(LT5),1) = 5;
LifetimesFull(1+length(LT1)+length(LT2)+length(LT3)+length(LT4)+length(LT5):length(LT1)+length(LT2)+length(LT3)+length(LT4)+length(LT5)+length(LT6),1) = 6;
LifetimesFull(1+length(LT1)+length(LT2)+length(LT3)+length(LT4)+length(LT5)+length(LT6):end,1) = 7;

csvwrite('LifetimesFull.csv',LifetimesFull);
