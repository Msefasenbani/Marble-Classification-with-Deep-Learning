clc;
clear all;
 
%LOAD DATA
digitDatasetPath = fullfile('D:\_MASAÜSTÜ\DATA_alexnet');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

%Split the data into 70% training and 30% test data
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');
 
%RANDOM IMAGES FROM DATA FOLDER
figure;
perm = randperm(100,20);
for i = 1:20
    subplot(4,5,i);
    imshow(imds.Files{perm(i)});
end
 



layers = [
    imageInputLayer([227 227 3],"Name","data")
    convolution2dLayer([11 11],96,"Name","conv1","BiasLearnRateFactor",2,"Stride",[4 4])
    reluLayer("Name","relu1")
    crossChannelNormalizationLayer(5,"Name","norm1","K",1)
    maxPooling2dLayer([3 3],"Name","pool1","Stride",[2 2])
    groupedConvolution2dLayer([5 5],128,2,"Name","conv2","BiasLearnRateFactor",2,"Padding",[2 2 2 2])
    reluLayer("Name","relu2")
    crossChannelNormalizationLayer(5,"Name","norm2","K",1)
    maxPooling2dLayer([3 3],"Name","pool2","Stride",[2 2])
    convolution2dLayer([3 3],384,"Name","conv3","BiasLearnRateFactor",2,"Padding",[1 1 1 1])
    reluLayer("Name","relu3")
    groupedConvolution2dLayer([3 3],192,2,"Name","conv4","BiasLearnRateFactor",2,"Padding",[1 1 1 1])
    reluLayer("Name","relu4")
    groupedConvolution2dLayer([3 3],128,2,"Name","conv5","BiasLearnRateFactor",2,"Padding",[1 1 1 1])
    reluLayer("Name","relu5")
    maxPooling2dLayer([3 3],"Name","pool5","Stride",[2 2])
    fullyConnectedLayer(4096,"Name","fc6","BiasLearnRateFactor",2)
    reluLayer("Name","relu6")
    dropoutLayer(0.5,"Name","drop6")
    fullyConnectedLayer(4096,"Name","fc7","BiasLearnRateFactor",2)
    reluLayer("Name","relu7")
    dropoutLayer(0.5,"Name","drop7")
    fullyConnectedLayer(5,"Name","fc","BiasLearnRateFactor",2)
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];

inputSize = layers(1).InputSize

augmenter = imageDataAugmenter( ...
    'RandXReflection' , true , ...
    'RandYReflection' , true );


%resize the training and test images and operations to perform on the training images
augimdsTrain      = augmentedImageDatastore([inputSize(1),inputSize(2),3],imdsTrain,'DataAugmentation',augmenter);
augimdsValidation = augmentedImageDatastore([inputSize(1),inputSize(2),3],imdsValidation);
 
%Training Options
options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',50, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');

%Train Model
net = trainNetwork(augimdsTrain,layers,options);

%Validation Accuracy
[YPred,scores] = classify(net,augimdsValidation);
accuracy = mean(YPred == imdsValidation.Labels)

%Confusion Matrix
plotconfusion(imdsValidation.Labels,YPred)

%Predict random images
idx = randperm(numel(imdsValidation.Files),16);
figure
for i = 1:16
    subplot(4,4,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label) + ", " + num2str(100*max(scores(idx(i),:)),3) + "%");
end
 
%Save Model
Alexnet_Model_1 = net;
save Alexnet_Model_1




