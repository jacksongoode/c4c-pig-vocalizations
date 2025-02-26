%% Create spectrograms for all labeled and sorted audio files
clear
clc
close all


tic
dataFolder = 'C:\Users\ccsyp\Documents\KU\Organized Vocals\INRA\Callable'
ads = audioDatastore(fullfile(dataFolder), ...
    'IncludeSubfolders',true, ...
    'FileExtensions','.wav', ...
    'LabelSource','foldernames')

window=0.03*44100;
noverlap=floor(0.99*window);
nfft=512;

for i = 1:numel(ads.Files)
    x = audioread(ads.Files{i});
    if length(x)/44100 <= 4.5
        xPadded = [zeros(floor((4.5*44100-size(x,1))/2),1);...
                    x;...
                    zeros(ceil((4.5*44100-size(x,1))/2),1)];

        spectrogram(xPadded,window,noverlap,nfft,44100,'yaxis');
        ylim([0 8]) %kHz
        xlim([0 4.5]) %s
        caxis([-160 -20])
        set(gca, 'Visible', 'off');
        colorbar('off');
    else
    end
        filename=strcat(ads.Files{i},'_spec.png');
        saveas(gcf,filename);
end
toc
