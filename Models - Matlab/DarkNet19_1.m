clc;
clear all;
 
%LOAD DATA
digitDatasetPath = fullfile('D:\_MASAÜSTÜ\2020 _BAHAR\Deep learning\DATA_299x299');
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
    imageInputLayer([256 256 3],"Name","input","Normalization","rescale-zero-one")
    convolution2dLayer([3 3],32,"Name","conv1","Padding","same")
    batchNormalizationLayer("Name","batchnorm1")
    leakyReluLayer(0.1,"Name","leaky1")
    maxPooling2dLayer([2 2],"Name","pool1","Stride",[2 2])
    convolution2dLayer([3 3],64,"Name","conv2","Padding","same")
    batchNormalizationLayer("Name","batchnorm2")
    leakyReluLayer(0.1,"Name","leaky2")
    maxPooling2dLayer([2 2],"Name","pool2","Stride",[2 2])
    convolution2dLayer([3 3],128,"Name","conv3","Padding","same")
    batchNormalizationLayer("Name","batchnorm3")
    leakyReluLayer(0.1,"Name","leaky3")
    convolution2dLayer([1 1],64,"Name","conv4","Padding","same")
    batchNormalizationLayer("Name","batchnorm4")
    leakyReluLayer(0.1,"Name","leaky4")
    convolution2dLayer([3 3],128,"Name","conv5","Padding","same")
    batchNormalizationLayer("Name","batchnorm5")
    leakyReluLayer(0.1,"Name","leaky5")
    maxPooling2dLayer([2 2],"Name","pool3","Stride",[2 2])
    convolution2dLayer([3 3],256,"Name","conv6","Padding","same")
    batchNormalizationLayer("Name","batchnorm6")
    leakyReluLayer(0.1,"Name","leaky6")
    convolution2dLayer([1 1],128,"Name","conv7","Padding","same")
    batchNormalizationLayer("Name","batchnorm7")
    leakyReluLayer(0.1,"Name","leaky7")
    convolution2dLayer([3 3],256,"Name","conv8","Padding","same")
    batchNormalizationLayer("Name","batchnorm8")
    leakyReluLayer(0.1,"Name","leaky8")
    maxPooling2dLayer([2 2],"Name","pool4","Stride",[2 2])
    convolution2dLayer([3 3],512,"Name","conv9","Padding","same")
    batchNormalizationLayer("Name","batchnorm9")
    leakyReluLayer(0.1,"Name","leaky9")
    convolution2dLayer([1 1],256,"Name","conv10","Padding","same")
    batchNormalizationLayer("Name","batchnorm10")
    leakyReluLayer(0.1,"Name","leaky10")
    convolution2dLayer([3 3],512,"Name","conv11","Padding","same")
    batchNormalizationLayer("Name","batchnorm11")
    leakyReluLayer(0.1,"Name","leaky11")
    convolution2dLayer([1 1],256,"Name","conv12","Padding","same")
    batchNormalizationLayer("Name","batchnorm12")
    leakyReluLayer(0.1,"Name","leaky12")
    convolution2dLayer([3 3],512,"Name","conv13","Padding","same")
    batchNormalizationLayer("Name","batchnorm13")
    leakyReluLayer(0.1,"Name","leaky13")
    maxPooling2dLayer([2 2],"Name","pool5","Stride",[2 2])
    convolution2dLayer([3 3],1024,"Name","conv14","Padding","same")
    batchNormalizationLayer("Name","batchnorm14")
    leakyReluLayer(0.1,"Name","leaky14")
    convolution2dLayer([1 1],512,"Name","conv15","Padding","same")
    batchNormalizationLayer("Name","batchnorm15")
    leakyReluLayer(0.1,"Name","leaky15")
    convolution2dLayer([3 3],1024,"Name","conv16","Padding","same")
    batchNormalizationLayer("Name","batchnorm16")
    leakyReluLayer(0.1,"Name","leaky16")
    convolution2dLayer([1 1],512,"Name","conv17","Padding","same")
    batchNormalizationLayer("Name","batchnorm17")
    leakyReluLayer(0.1,"Name","leaky17")
    convolution2dLayer([3 3],1024,"Name","conv18","Padding","same")
    batchNormalizationLayer("Name","batchnorm18")
    leakyReluLayer(0.1,"Name","leaky18")
    convolution2dLayer([3 3],5,"Name","conv","Padding","same")
    globalAveragePooling2dLayer("Name","avg1")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","output")];

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
DarkNet19_Model_1 = net;
save DarkNet19_Model_1




