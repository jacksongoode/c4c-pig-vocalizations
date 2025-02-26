function [net] = NN_prep_for_classify(net,augvalset,augtrainingset)

lgraph = layerGraph(net);

% Retrain the NN
valfrequency = floor((numel(augtrainingset.Files)/32)); % validation test frequency
options = trainingOptions('sgdm', ...
    'MiniBatchSize',32, ...
    'MaxEpochs',1, ... %  An epoch is a full training cycle on the entire training data set
    'InitialLearnRate',1e-6, ... %1e-6 smallest % the most important parameter, can only be discovered thru trial and error
    'Shuffle','every-epoch', ... % important to shuffle up the mini batches
    'ValidationData',augvalset, ...
    'ValidationFrequency',valfrequency, ...
    'Verbose',false);
   
[net] = trainNetwork(augtrainingset,lgraph,options);
end

