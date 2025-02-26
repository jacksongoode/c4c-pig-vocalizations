function [augimds] = SiteValidationIMDS(Files,Labels)

% Import spectrograms
files=string(Files);
for counter = 1:length(files)
    fullfilenames(counter) = fullfile('C:\Users\ccsyp\Documents\KU\Properly Renamed Vocals',files(counter));
end

fullfilenames=fullfilenames';
imds=imageDatastore(fullfilenames);
imds.Labels = Labels;
augimds = augmentedImageDatastore([224 224 3], imds);
end

