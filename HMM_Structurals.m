%% Analyses of structural connectome properties in our sample

struct = load('REDT2_qsiprepmat.mat');
connectomes = struct.REDT2_qsiprep.schaefer100x7.connectivity;
connectomes = squeeze(connectomes(:,1,:));
connectomes = reshape(connectomes,[57,100,100]);
ids = struct.REDT2_qsiprep.sample.id.sub;




