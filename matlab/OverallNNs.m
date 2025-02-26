tic
%% Initialize
% Import data
cd 'C:\Users\ccsyp\Documents\KU'
[~,txtData]  = xlsread('MasterSheet_sortedbysite.xlsx');
ContextCatb = categorical(txtData(:,17));
Files = txtData(:,12);
Valence = categorical(txtData(:,20));
Type = categorical(txtData(:,39));
Site = categorical(txtData(:,2));

CheckpointPathVal = {'C:\Users\ccsyp\Documents\KU\Codes\Manuscript Set\OverAllValCPFolder1',...
    'C:\Users\ccsyp\Documents\KU\Codes\Manuscript Set\OverAllValCPFolder2',...
    'C:\Users\ccsyp\Documents\KU\Codes\Manuscript Set\OverAllValCPFolder3',...
    'C:\Users\ccsyp\Documents\KU\Codes\Manuscript Set\OverAllValCPFolder4',...
    'C:\Users\ccsyp\Documents\KU\Codes\Manuscript Set\OverAllValCPFolder5',...
    'C:\Users\ccsyp\Documents\KU\Codes\Manuscript Set\OverAllValCPFolder6',...
    'C:\Users\ccsyp\Documents\KU\Codes\Manuscript Set\OverAllValCPFolder7',...
    'C:\Users\ccsyp\Documents\KU\Codes\Manuscript Set\OverAllValCPFolder8',...
    'C:\Users\ccsyp\Documents\KU\Codes\Manuscript Set\OverAllValCPFolder9',...
    'C:\Users\ccsyp\Documents\KU\Codes\Manuscript Set\OverAllValCPFolder10',...
    'C:\Users\ccsyp\Documents\KU\Codes\Manuscript Set\OverAllValCPFolder11',...
    'C:\Users\ccsyp\Documents\KU\Codes\Manuscript Set\OverAllValCPFolder12'};
CheckpointPathCon = {'C:\Users\ccsyp\Documents\KU\Codes\Manuscript Set\OverAllConCPFolder1',...
    'C:\Users\ccsyp\Documents\KU\Codes\Manuscript Set\OverAllConCPFolder2',...
    'C:\Users\ccsyp\Documents\KU\Codes\Manuscript Set\OverAllConCPFolder3',...
    'C:\Users\ccsyp\Documents\KU\Codes\Manuscript Set\OverAllConCPFolder4',...
    'C:\Users\ccsyp\Documents\KU\Codes\Manuscript Set\OverAllConCPFolder5',...
    'C:\Users\ccsyp\Documents\KU\Codes\Manuscript Set\OverAllConCPFolder6',...
    'C:\Users\ccsyp\Documents\KU\Codes\Manuscript Set\OverAllConCPFolder7',...
    'C:\Users\ccsyp\Documents\KU\Codes\Manuscript Set\OverAllConCPFolder8',...
    'C:\Users\ccsyp\Documents\KU\Codes\Manuscript Set\OverAllConCPFolder9',...
    'C:\Users\ccsyp\Documents\KU\Codes\Manuscript Set\OverAllConCPFolder10',...
    'C:\Users\ccsyp\Documents\KU\Codes\Manuscript Set\OverAllConCPFolder11',...
    'C:\Users\ccsyp\Documents\KU\Codes\Manuscript Set\OverAllConCPFolder12'};
site_labels = {'ETHZ', 'FBN', 'IASPA', 'IASPB', 'IASPC', 'INRA', 'NMBU'};
featureLayer = 'avg_pool';

%% Overall Networks

for i = 1:12

    display('Beginning Loop:')
    display(i)

    % Train NN on valences
    cd 'C:\Users\ccsyp\Documents\KU\Codes\Manuscript Set'
    [val_perf,val_trainingset,val_valset,val_valset_Labels,val_valLabelCount,augimds] = NN_function(Files,Valence,1,32,Inf,CheckpointPathVal{i});
    
    % Train NN on contexts
    cd 'C:\Users\ccsyp\Documents\KU\Codes\Manuscript Set'
    [con_perf,con_trainingset,con_valset,con_valset_Labels,con_valLabelCount,augimds] = NN_function(Files,ContextCatb,0,32,Inf,CheckpointPathCon{i});
    
    % save training set and validation set with the checkpoints for this run
    save(fullfile(CheckpointPathVal{i},'TrainingSet.mat'),'val_trainingset')
    save(fullfile(CheckpointPathVal{i},'ValidationSet.mat'),'val_valset')
    
    save(fullfile(CheckpointPathCon{i},'TrainingSet.mat'),'con_trainingset')
    save(fullfile(CheckpointPathCon{i},'ValidationSet.mat'),'con_valset')

    % find highest validation accuracy in perf and which iteration it occurs at
    [val_peak_acc, val_iter] = max(val_perf.ValidationAccuracy);
    [con_peak_acc, con_iter] = max(con_perf.ValidationAccuracy);
    
    % use wildcard to load NN at that checkpoint
    fname = dir(fullfile(CheckpointPathVal{i},sprintf('%s%i%s%i', 'net_checkpoint__', val_iter,'__*.mat')));
    load(fullfile(CheckpointPathVal{i},fname(1).name),'net');
    val_net = net;
    
    fname = dir(fullfile(CheckpointPathCon{i},sprintf('%s%i%s%i', 'net_checkpoint__', con_iter,'__*.mat')));
    load(fullfile(CheckpointPathCon{i},fname(1).name),'net');
    con_net = net;
    
    % use net to classify validation set and make confusion matrix
    cd 'C:\Users\ccsyp\Documents\KU\Codes\Manuscript Set'
    [val_net] = NN_prep_for_classify(val_net,val_valset,val_trainingset);
    val_predictions = classify(val_net,val_valset);
    val_labels = val_valset_Labels;
    val_OVERALL_accuracy(i) = sum(val_predictions == val_labels)/numel(val_labels);
    val_conf = confusionmat(val_labels,val_predictions);
%     confusionchart(val_labels,val_predictions)
    
    [con_net] = NN_prep_for_classify(con_net,con_valset,con_trainingset);
    con_predictions = classify(con_net,con_valset);
    con_labels = con_valset_Labels;
    con_OVERALL_accuracy(i) = sum(con_predictions == con_labels)/numel(con_labels);
    con_conf = confusionmat(con_labels,con_predictions);
    
    % use confusion matrix to calculate recall, precision, F1 
    [val_precision,val_recall,val_F1,val_weighted_precision,val_weighted_recall,val_weighted_F1] = F1(val_conf,val_valLabelCount);
    [con_precision,con_recall,con_F1,con_weighted_precision,con_weighted_recall,con_weighted_F1] = F1(con_conf,con_valLabelCount);
    
    % save all these
    val_prec_rec_F1{i} = [val_precision val_recall val_F1];
    save(fullfile(CheckpointPathVal{i},'Overall_Valence_NN_PrecRecF1Vectors.mat'),'val_prec_rec_F1');
    val_metrics{i} = [val_OVERALL_accuracy val_weighted_precision val_weighted_recall val_weighted_F1];
    save(fullfile(CheckpointPathVal{i},'Overall_Valence_NN_WeightedMetrics.mat'),'val_metrics');
    
    con_prec_rec_F1{i} = [con_precision con_recall con_F1];
    save(fullfile(CheckpointPathCon{i},'Overall_Context_NN_PrecRecF1Vectors.mat'),'con_prec_rec_F1');
    con_metrics{i} = [con_OVERALL_accuracy con_weighted_precision con_weighted_recall con_weighted_F1];
    save(fullfile(CheckpointPathCon{i},'Overall_Context_NN_WeightedMetrics.mat'),'con_metrics');
   
    %% Use Overall NN to classify valence within each site
    valFiles = val_valset.Files;
    for k = 1:length(valFiles)
        cellContents = valFiles{k};
        % Truncate and stick back into the cell
        valFiles{k} = cellContents(53:end);
    end
    valencevalidationsetindices = ismember(Files,valFiles);
    conFiles = con_valset.Files;
    for k = 1:length(conFiles)
        cellContents = conFiles{k};
        % Truncate and stick back into the cell
        conFiles{k} = cellContents(53:end);
    end
    contextvalidationsetindices = ismember(Files,conFiles);

    % Use Overall NN to classify valence within each site
    for j = 1:length(site_labels)

        siteindex = ismember(Site,site_labels{j});
        siteandvalidationintersect = intersect(find(valencevalidationsetindices),find(siteindex));
        siteandcontextintersect = intersect(find(contextvalidationsetindices),find(siteindex));

        % Valence Accuracy on Validation Set:
        [val_imds] = SiteValidationIMDS(Files(siteandvalidationintersect),Valence(siteandvalidationintersect));
        val_predictions = classify(val_net,val_imds);
        val_labels = Valence(siteandvalidationintersect);
        val_winSITE_accuracy{i,j} = sum(val_predictions == val_labels)/numel(val_labels);

        % Context Accuracy on Validation Set:
        [con_imds] = SiteValidationIMDS(Files(siteandcontextintersect),ContextCatb(siteandcontextintersect));
        con_predictions = classify(con_net,con_imds);
        con_labels = ContextCatb(siteandcontextintersect);
        con_winSITE_accuracy{i,j} = sum(con_predictions == con_labels)/numel(con_labels);
    end
    % Save within site classifier accuracies
    save(fullfile(CheckpointPathVal{i},'OverallNN_Within_Site_Accuracies.mat'),'val_winSITE_accuracy')
    save(fullfile(CheckpointPathCon{i},'OverallNN_Within_Site_Accuracies.mat'),'con_winSITE_accuracy')
    
    
    % % incomplete idea, needs to be checked
    % calculate features for current highest accuracy network
    % if i==1
    %     val_features = activations(val_net, augimds, featureLayer, ...
    %         'MiniBatchSize', 32, 'OutputAs', 'columns');
    %     val_features = val_features';
    % elseif val_OVERALL_accuracy(i) > val_OVERALL_accuracy(i-1)
    %     val_features = activations(val_net, augimds, featureLayer, ...
    %         'MiniBatchSize', 32, 'OutputAs', 'columns');
    %     val_features = val_features';
    %     cd 'C:\Users\ccsyp\Documents\KU\Codes\Manuscript Set'
    %     save('Features_From_Valence_OVERALL.mat','val_features')
    % else
    % end
    % if i==1
    %     con_features = activations(con_net, augimds, featureLayer, ...
    %         'MiniBatchSize', 32, 'OutputAs', 'columns');
    %     con_features = con_features';
    % elseif con_OVERALL_accuracy(i) > con_OVERALL_accuracy(i-1)
    %     con_features = activations(con_net, augimds, featureLayer, ...
    %         'MiniBatchSize', 32, 'OutputAs', 'columns');
    %     con_features = con_features';
    %     % save features
    %     cd 'C:\Users\ccsyp\Documents\KU\Codes\Manuscript Set'
    %     save('Features_From_Context_OVERALL.mat','con_features')
    % else
    % end
    toc
end
