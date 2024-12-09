%{
  make_fig18.m
  
  Reproduces Figures 18 in O'Shaughnessy et al., 'Generative causal
  explanations of black-box classifiers,' Proc. NeurIPS 2020: final
  value of causal effect and data fidelity terms in objective for
  various capacities of VAE.
  
  Note: this plots the output of make_fig18_fig19.py, ./results/fig18.mat.
  This script requires the cbrewer package:
  https://www.mathworks.com/matlabcentral/fileexchange/34087-cbrewer-colorbrewer-schemes-for-matlab
%}

%%
load results/fig18.mat

lambda = logspace(-3,-1,10);
filts = [4 8 16 32 48 64];
nsteps = 50;

data.loss_ce = data.loss_ce(2:end,:,:);
data.loss_nll = data.loss_nll(2:end,:,:);
filts = filts(2:end);

%%
leg = cell(1,length(filts));
for i = 1:length(filts)
  leg{i} = sprintf('%d',filts(i));
end

figure(1); clf;
imagesc(lambda,filts,mean(-data.loss_ce(:,:,end-nsteps+1:end),3));
xlabel('\lambda'); ylabel('filters/layer'); title('C'); colorbar;

figure(2); clf;
imagesc(lambda,filts,mean(-data.loss_nll(:,:,end-nsteps+1:end),3)./lambda);
xlabel('\lambda'); ylabel('filters/layer'); title('D'); colorbar;

%%
figure(3); clf;
cols = cbrewer('seq','Blues',length(filts)+2);
for i = 1:length(filts)
  semilogx(lambda, -mean(data.loss_ce(i,:,end-nsteps+1:end),3), ...
    'color',cols(i+2,:)); hold on;
end
semilogx(lambda, ones(size(lambda))*log2(3)*log(2), 'k--');
grid on; ylim([0.9 1.11]); set(gca,'fontsize',12);
xlabel('$$\lambda$$','interpreter','latex');
title('$$\mathcal{C}$$','interpreter','latex');
hleg = legend(leg,'location','sw'); set(gca,'fontname','Times New Roman');
title(hleg, 'Filters/layer'); set(gcf,'color','white');
export_fig figs/fig18a.pdf

figure(4); clf;
cols = cbrewer('seq','Blues',length(filts)+2);
for i = 1:length(filts)
  semilogx(lambda, -mean(data.loss_nll(i,:,end-nsteps+1:end),3)./lambda, ...
    'color',cols(i+2,:)); hold on;
end
grid on; set(gca,'fontsize',12);
xlabel('$$\lambda$$','interpreter','latex');
title('$$\mathcal{D}$$','interpreter','latex');
hleg = legend(leg,'location','se'); set(gca,'fontname','Times New Roman');
title(hleg, 'Filters/layer'); set(gcf,'color','white');
export_fig figs/fig18b.pdf