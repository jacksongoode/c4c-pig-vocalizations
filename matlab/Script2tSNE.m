clear
clc
close all
%% t-SNE Applied to Max F1-Scoring NN Checkpoint
% Import data
cd 'C:\Users\ccsyp\Documents\KU'
[~,txtData]  = xlsread('NewMasterSheet.xlsx');
ContextCatb = categorical(txtData(:,7));
Files = txtData(:,6);
Valence = categorical(txtData(:,8));
Site = categorical(txtData(:,2));

cd 'C:\Users\ccsyp\Documents\KU\Codes\Manuscript Set\INRA Redo'
% load('Features_From_Valence_OVERALL.mat','val_features')
load('Features_From_Context_OVERALL.mat','con_features')

% Apply t-SNE to features (try using 'Perplexity' [10, 15, 20, 25, 30, 35,
% 40, 45, 50])
% for i = [10, 15, 20, 25, 30, 35, 40, 45, 50]
%     rng(2)
%     val_tsne=tsne(val_features,'Perplexity',i);
%     figure
%     gscatter(val_tsne(:,1),val_tsne(:,2),Valence)
%     title(i)
% end
% 
% for i = [10, 15, 20, 25, 30, 35, 40, 45, 50]
%     rng(2)
%     con_tsne=tsne(con_features,'Perplexity',i);
%     figure
%     gscatter(con_tsne(:,1),con_tsne(:,2),ContextCatb)
%     title(i)
% end
% 
% % choose which perplexity to continue with
% display('whatcha think?')
% val_perp = input('Which perplexity value would you like to use for the valence t-SNE?')
% con_perp = input('Which perplexity value would you like to use for the CONTEXT t-SNE?')

%% plot using transparent dots
% % % % % rng(2)
% % % % % val_tsne=tsne(val_features,'Perplexity',50);
% % % % % 
% % % % % figure
% % % % % neg = scatter(val_tsne(Valence=='Neg',1),val_tsne(Valence=='Neg',2),20,'filled','r');
% % % % % neg.MarkerFaceAlpha = 0.6;
% % % % % neg.Marker = '^';
% % % % % hold on
% % % % % pos = scatter(val_tsne(Valence=='Pos',1),val_tsne(Valence=='Pos',2),20,'filled','g');
% % % % % pos.MarkerFaceAlpha = 0.6;
% % % % % pos.Marker = 'o';
% % % % % hold on
% % % % % legend('Negative','Positive')

%%
% for i = [15]
%      rng(2)
%      con_tsne=tsne(con_features,'Perplexity',5);
%      figure;
%      gscatter(con_tsne(:,1),con_tsne(:,2),ContextCatb)
%      rng(2)
%      con_tsne=tsne(con_features,'Perplexity',10);
%      figure;
%      gscatter(con_tsne(:,1),con_tsne(:,2),ContextCatb)
     
     rng(4)
     con_tsne=tsne(con_features,'Perplexity',20);
     figure;
     gscatter(con_tsne(:,1),con_tsne(:,2),ContextCatb)
%      figure;
%      gscatter(con_tsne(:,1),con_tsne(:,2),Site)
     rng(4)
     con_tsne=tsne(con_features,'Perplexity',25);
     figure;
     gscatter(con_tsne(:,1),con_tsne(:,2),ContextCatb)
%      figure;
%      gscatter(con_tsne(:,1),con_tsne(:,2),Site)
     rng(4)
     con_tsne=tsne(con_features,'Perplexity',30);
     figure;
     gscatter(con_tsne(:,1),con_tsne(:,2),ContextCatb)
%      figure;
%      gscatter(con_tsne(:,1),con_tsne(:,2),Site)
     
%     cd 'C:\Users\ccsyp\Documents\KU\Codes\Manuscript Set'
%     contextlabels = {'AfterNursing','Barren','BeforeNursing','Castration','Crushing','Enriched','Fighting','Handling','Huddling','Isolation','MissedNursing','NegativeCondtioning','NovelObject','PositiveConditioning','Restrain','Reunion','Run','Surprise','Waiting'};
% 
%     colors = maxdistcolor(length(contextlabels),@sRGB_to_CIELab);
%     
%     figure
%     for j = 1:length(contextlabels)
%         confortry = scatter(con_tsne(ContextCatb==contextlabels{j},1),con_tsne(ContextCatb==contextlabels{j},2),20,'filled');
%         confortry.MarkerFaceAlpha = 0.6;
%         confortry.MarkerFaceColor = colors(j,:);
%         hold on
%     end
%     legend('AfterNursing','Barren','BeforeNursing','Castration','Crushing','Enriched','Fighting','Huddling','Isolation','MissedNursing','NegativeCondtioning','NovelObject','PositiveConditioning','Restrain','Reunion','Run','Slaughterhouse','Surprise')
% 
%     
%     figure
%     siteplot1 = scatter(con_tsne(Site=='ETHZ',1),con_tsne(Site=='ETHZ',2),20,'filled');
%     siteplot1.Marker = 'o';
%     siteplot1.MarkerFaceAlpha = 0.6;
%     hold on
%     siteplot2 = scatter(con_tsne(Site=='FBN',1),con_tsne(Site=='FBN',2),20,'filled');
%     siteplot2.Marker = 'o';
%     siteplot2.MarkerFaceAlpha = 0.6;
%     hold on
%     siteplot3 = scatter(con_tsne(Site=='IASPA',1),con_tsne(Site=='IASPA',2),20,'filled');
%     siteplot3.Marker = 'o';
%     siteplot3.MarkerFaceAlpha = 0.6;
%     hold on
%     siteplot4 = scatter(con_tsne(Site=='IASPB',1),con_tsne(Site=='IASPB',2),20,'filled');
%     siteplot4.Marker = 'o';
%     siteplot4.MarkerFaceAlpha = 0.6;
%     hold on
%     siteplot5 = scatter(con_tsne(Site=='IASPC',1),con_tsne(Site=='IASPC',2),20,'filled');
%     siteplot5.Marker = 'o';
%     siteplot5.MarkerFaceAlpha = 0.6;
%     hold on
%     siteplot6 = scatter(con_tsne(Site=='INRA',1),con_tsne(Site=='INRA',2),20,'filled');
%     siteplot6.Marker = 'o';
%     siteplot6.MarkerFaceAlpha = 0.6;
%     hold on
%     siteplot7 = scatter(con_tsne(Site=='NMBU',1),con_tsne(Site=='NMBU',2),20,'filled');
%     siteplot7.Marker = 'o';
%     siteplot7.MarkerFaceAlpha = 0.6;
%     legend('ETHZ','FBN','IASPA','IASPB','IASPC','INRA','NMBU')
%     
% end

%% plotting t-SNE with better markers
contextlabels = {'AfterNursing','Barren','BeforeNursing','Castration','Crushing','Enriched','Fighting','Handling','Huddling','Isolation','MissedNursing','NegativeCondtioning','NovelObject','PositiveConditioning','Restrain','Reunion','Run','Surprise','Waiting'};
poscons = {'AfterNursing','BeforeNursing','Enriched','Huddling','PositiveConditioning','Reunion','Run'};
cd 'C:\Users\ccsyp\Documents\KU\Codes\Manuscript Set'
neededcolors = [0.111111111000000,0.755905512000000,0.349206349000000;0.777777778000000,0.748031496000000,0.396825397000000;0.0634920630000000,0.503937008000000,0.571428571000000;1,0,0.714285714000000;0.904761905000000,1,0;0.0476190480000000,0.590551181000000,1;0.932000000000000,0.600000000000000,0.488000000000000;0,1,0.616000000000000;0.714285714000000,0.0472440940000000,0.269841270000000;0.556000000000000,0,0.556000000000000;0.857142857000000,0.559055118000000,0.793650794000000;1,0.582677165000000,0;0,1,0.936507937000000;0.666666667000000,0.393700787000000,1;0,0,1;0,1,0;1,0,0.0476190480000000;0,0.00787401600000000,0.365079365000000];
colors = maxdistcolor(length(contextlabels),@sRGB_to_CIELab,'inc', neededcolors);
% hexcolors = ['jjjjjjj','#2f4f4f','#7f0000','#008000','#8b008b','#ff4500','#ffa500','#ffff00','#40e0d0','#7fff00','#00fa9a','#4169e1','#e9967a','#00bfff','#0000ff','#ff00ff','#f0e68c','#dda0dd','#ff1493'];

figure
for j = 1:length(contextlabels)
    if ismember(contextlabels{j},poscons)
        confortry = scatter(con_tsne(ContextCatb==contextlabels{j},1),con_tsne(ContextCatb==contextlabels{j},2),20,'filled');
        confortry.Marker = 'o';
        confortry.MarkerFaceAlpha = 0.6;
        confortry.MarkerFaceColor = colors(j,:);
        hold on
    else
        confortry = scatter(con_tsne(ContextCatb==contextlabels{j},1),con_tsne(ContextCatb==contextlabels{j},2),20,'filled');
        confortry.Marker = '^';
        confortry.MarkerFaceAlpha = 0.6;
        confortry.MarkerFaceColor = colors(j,:);
        hold on
    end
    hold on
    axis([-85 85 -85 85])
    hold on
end
legend('AfterNursing','Barren','BeforeNursing','Castration','Crushing','Enriched','Fighting','Huddling','Isolation','MissedNursing','NegativeConditioning','NovelObject','PositiveConditioning','Restrain','Reunion','Running','Slaughterhouse','Surprise')

figure
siteplot1 = scatter(con_tsne(Site=='ETHZ',1),con_tsne(Site=='ETHZ',2),20,'filled');
siteplot1.Marker = 'o';
siteplot1.MarkerFaceAlpha = 0.6;
hold on
siteplot2 = scatter(con_tsne(Site=='FBN',1),con_tsne(Site=='FBN',2),20,'filled');
siteplot2.Marker = 'o';
siteplot2.MarkerFaceAlpha = 0.6;
hold on
siteplot3 = scatter(con_tsne(Site=='IASPA',1),con_tsne(Site=='IASPA',2),20,'filled');
siteplot3.Marker = 'o';
siteplot3.MarkerFaceAlpha = 0.6;
hold on
siteplot4 = scatter(con_tsne(Site=='IASPB',1),con_tsne(Site=='IASPB',2),20,'filled');
siteplot4.Marker = 'o';
siteplot4.MarkerFaceAlpha = 0.6;
hold on
siteplot5 = scatter(con_tsne(Site=='IASPC',1),con_tsne(Site=='IASPC',2),20,'filled');
siteplot5.Marker = 'o';
siteplot5.MarkerFaceAlpha = 0.6;
hold on
siteplot6 = scatter(con_tsne(Site=='INRA',1),con_tsne(Site=='INRA',2),20,'filled');
siteplot6.Marker = 'o';
siteplot6.MarkerFaceAlpha = 0.6;
hold on
siteplot7 = scatter(con_tsne(Site=='NMBU',1),con_tsne(Site=='NMBU',2),20,'filled');
siteplot7.Marker = 'o';
siteplot7.MarkerFaceAlpha = 0.6;
hold on
axis([-85 85 -85 85])
legend('ETHZ','FBN','IASPA','IASPB','IASPC','INRA','NMBU')