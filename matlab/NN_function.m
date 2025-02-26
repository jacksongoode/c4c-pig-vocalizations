function [performance,augtrainingset,augvalset,valset_Labels,valLabelCount,augimds] = NN_function(Files,Labels,EqualizeLabels,MiniBatchSize,ValidationPatience,CheckpointPath)

% Import spectrograms
cd 'C:\Users\ccsyp\Documents\KU\Properly Renamed Vocals - Copy with INRA Contexts'
files=string(Files);
for i = 1:length(files)
    fullfilenames(i) = fullfile('Properly Renamed Vocals - Copy with INRA Contexts',files(i));
end

fullfilenames=fullfilenames';
imds=imageDatastore(fullfilenames);
imds.Labels = Labels;
tbl = countEachLabel(imds)

% Load pretrained network
net = resnet50();
imagesize = net.Layers(1).InputSize;
augimds = augmentedImageDatastore(imagesize, imds);

if EqualizeLabels==1
    % Determine the smallest amount of images in a category
    minSetCount = min(tbl{:,2});

    % Use splitEachLabel method to trim the set
    [imds] = splitEachLabel(imds, minSetCount, 'randomize');

    % Notice that each set now has exactly the same number of images
    countEachLabel(imds)
else
end

% Use split again to divide into training, validation, and testing sets
% (for generalizability measure)
[trainingset, valset] = splitEachLabel(imds, 0.7, 'randomize');
valset_Labels = valset.Labels;
countEachLabel(trainingset)
valLabelCount = countEachLabel(valset)

% Resize images for NN
augvalset = augmentedImageDatastore(imagesize,valset);
augtrainingset = augmentedImageDatastore(imagesize,trainingset);

% Replace last few layers, primed for 1000s of classes
lgraph = layerGraph(net);

cd 'C:\Users\ccsyp\Documents\KU\Codes'
[learnablelayer,classlayer] = findLayersToReplace(lgraph);

% In most networks, the last layer with learnable weights is a fully connected layer.
% Replace this fully connected layer with a new fully connected layer with
% the number of outputs equal to the number of classes in the new data set.
numclasses = numel(categories(trainingset.Labels));

newLearnableLayer = fullyConnectedLayer(numclasses, ...
    'Name','new_fc', ...
    'WeightLearnRateFactor',10, ...
    'BiasLearnRateFactor',10);

lgraph = replaceLayer(lgraph,learnablelayer.Name,newLearnableLayer);

% The classification layer specifies the output classes of the network.
% Replace the classification layer with a new one without class labels.
% trainNetwork automatically sets the output classes of the layer at training time.
newclasslayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classlayer.Name,newclasslayer);


% Retrain the NN
valfrequency = floor((numel(augtrainingset.Files)/MiniBatchSize)); % validation test frequency
options = trainingOptions('sgdm', ...
    'MiniBatchSize',MiniBatchSize, ...
    'MaxEpochs',20, ... %  An epoch is a full training cycle on the entire training data set
    'InitialLearnRate',0.001, ... %1e-6 smallest % the most important parameter, can only be discovered thru trial and error
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',5, ...
    'LearnRateDropFactor',10^-0.5,...
    'Shuffle','every-epoch', ... % important to shuffle up the mini batches
    'ValidationData',augvalset, ...
    'ValidationFrequency',valfrequency, ...
    'ValidationPatience', ValidationPatience, ... % if the loss function increases for 10 epochs, implement "early stopping"
    'Verbose',false, ...
    'CheckpointPath',CheckpointPath);
   
[net,performance] = trainNetwork(augtrainingset,lgraph,options);
end

