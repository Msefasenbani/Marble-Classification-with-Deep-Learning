clc;
clear all;
 
%LOAD DATA
digitDatasetPath = fullfile('D:\_MASAÜSTÜ\D');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

%Split the data into 70% training and 30% test data
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');
 
% %RANDOM IMAGES FROM DATA FOLDER
% figure;
% perm = randperm(100,20);
% for i = 1:20
%     subplot(4,5,i);
%     imshow(imds.Files{perm(i)});
% end

lgraph = layerGraph();


tempLayers = [
    imageInputLayer([224 224 3],"Name","input_1","Normalization","zscore")
    convolution2dLayer([3 3],32,"Name","Conv1","Padding","same","Stride",[2 2])
    batchNormalizationLayer("Name","bn_Conv1","Epsilon",0.001)
    clippedReluLayer(6,"Name","Conv1_relu")
    groupedConvolution2dLayer([3 3],1,32,"Name","expanded_conv_depthwise","Padding","same")
    batchNormalizationLayer("Name","expanded_conv_depthwise_BN","Epsilon",0.001)
    clippedReluLayer(6,"Name","expanded_conv_depthwise_relu")
    convolution2dLayer([1 1],16,"Name","expanded_conv_project","Padding","same")
    batchNormalizationLayer("Name","expanded_conv_project_BN","Epsilon",0.001)
    convolution2dLayer([1 1],96,"Name","block_1_expand","Padding","same")
    batchNormalizationLayer("Name","block_1_expand_BN","Epsilon",0.001)
    clippedReluLayer(6,"Name","block_1_expand_relu")
    groupedConvolution2dLayer([3 3],1,96,"Name","block_1_depthwise","Padding","same","Stride",[2 2])
    batchNormalizationLayer("Name","block_1_depthwise_BN","Epsilon",0.001)
    clippedReluLayer(6,"Name","block_1_depthwise_relu")
    convolution2dLayer([1 1],24,"Name","block_1_project","Padding","same")
    batchNormalizationLayer("Name","block_1_project_BN","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],144,"Name","block_2_expand","Padding","same")
    batchNormalizationLayer("Name","block_2_expand_BN","Epsilon",0.001)
    clippedReluLayer(6,"Name","block_2_expand_relu")
    groupedConvolution2dLayer([3 3],1,144,"Name","block_2_depthwise","Padding","same")
    batchNormalizationLayer("Name","block_2_depthwise_BN","Epsilon",0.001)
    clippedReluLayer(6,"Name","block_2_depthwise_relu")
    convolution2dLayer([1 1],24,"Name","block_2_project","Padding","same")
    batchNormalizationLayer("Name","block_2_project_BN","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","block_2_add")
    convolution2dLayer([1 1],144,"Name","block_3_expand","Padding","same")
    batchNormalizationLayer("Name","block_3_expand_BN","Epsilon",0.001)
    clippedReluLayer(6,"Name","block_3_expand_relu")
    groupedConvolution2dLayer([3 3],1,144,"Name","block_3_depthwise","Padding","same","Stride",[2 2])
    batchNormalizationLayer("Name","block_3_depthwise_BN","Epsilon",0.001)
    clippedReluLayer(6,"Name","block_3_depthwise_relu")
    convolution2dLayer([1 1],32,"Name","block_3_project","Padding","same")
    batchNormalizationLayer("Name","block_3_project_BN","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","block_4_expand","Padding","same")
    batchNormalizationLayer("Name","block_4_expand_BN","Epsilon",0.001)
    clippedReluLayer(6,"Name","block_4_expand_relu")
    groupedConvolution2dLayer([3 3],1,192,"Name","block_4_depthwise","Padding","same")
    batchNormalizationLayer("Name","block_4_depthwise_BN","Epsilon",0.001)
    clippedReluLayer(6,"Name","block_4_depthwise_relu")
    convolution2dLayer([1 1],32,"Name","block_4_project","Padding","same")
    batchNormalizationLayer("Name","block_4_project_BN","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","block_4_add");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","block_5_expand","Padding","same")
    batchNormalizationLayer("Name","block_5_expand_BN","Epsilon",0.001)
    clippedReluLayer(6,"Name","block_5_expand_relu")
    groupedConvolution2dLayer([3 3],1,192,"Name","block_5_depthwise","Padding","same")
    batchNormalizationLayer("Name","block_5_depthwise_BN","Epsilon",0.001)
    clippedReluLayer(6,"Name","block_5_depthwise_relu")
    convolution2dLayer([1 1],32,"Name","block_5_project","Padding","same")
    batchNormalizationLayer("Name","block_5_project_BN","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","block_5_add")
    convolution2dLayer([1 1],192,"Name","block_6_expand","Padding","same")
    batchNormalizationLayer("Name","block_6_expand_BN","Epsilon",0.001)
    clippedReluLayer(6,"Name","block_6_expand_relu")
    groupedConvolution2dLayer([3 3],1,192,"Name","block_6_depthwise","Padding","same","Stride",[2 2])
    batchNormalizationLayer("Name","block_6_depthwise_BN","Epsilon",0.001)
    clippedReluLayer(6,"Name","block_6_depthwise_relu")
    convolution2dLayer([1 1],64,"Name","block_6_project","Padding","same")
    batchNormalizationLayer("Name","block_6_project_BN","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],384,"Name","block_7_expand","Padding","same")
    batchNormalizationLayer("Name","block_7_expand_BN","Epsilon",0.001)
    clippedReluLayer(6,"Name","block_7_expand_relu")
    groupedConvolution2dLayer([3 3],1,384,"Name","block_7_depthwise","Padding","same")
    batchNormalizationLayer("Name","block_7_depthwise_BN","Epsilon",0.001)
    clippedReluLayer(6,"Name","block_7_depthwise_relu")
    convolution2dLayer([1 1],64,"Name","block_7_project","Padding","same")
    batchNormalizationLayer("Name","block_7_project_BN","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","block_7_add");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],384,"Name","block_8_expand","Padding","same")
    batchNormalizationLayer("Name","block_8_expand_BN","Epsilon",0.001)
    clippedReluLayer(6,"Name","block_8_expand_relu")
    groupedConvolution2dLayer([3 3],1,384,"Name","block_8_depthwise","Padding","same")
    batchNormalizationLayer("Name","block_8_depthwise_BN","Epsilon",0.001)
    clippedReluLayer(6,"Name","block_8_depthwise_relu")
    convolution2dLayer([1 1],64,"Name","block_8_project","Padding","same")
    batchNormalizationLayer("Name","block_8_project_BN","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","block_8_add");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],384,"Name","block_9_expand","Padding","same")
    batchNormalizationLayer("Name","block_9_expand_BN","Epsilon",0.001)
    clippedReluLayer(6,"Name","block_9_expand_relu")
    groupedConvolution2dLayer([3 3],1,384,"Name","block_9_depthwise","Padding","same")
    batchNormalizationLayer("Name","block_9_depthwise_BN","Epsilon",0.001)
    clippedReluLayer(6,"Name","block_9_depthwise_relu")
    convolution2dLayer([1 1],64,"Name","block_9_project","Padding","same")
    batchNormalizationLayer("Name","block_9_project_BN","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","block_9_add")
    convolution2dLayer([1 1],384,"Name","block_10_expand","Padding","same")
    batchNormalizationLayer("Name","block_10_expand_BN","Epsilon",0.001)
    clippedReluLayer(6,"Name","block_10_expand_relu")
    groupedConvolution2dLayer([3 3],1,384,"Name","block_10_depthwise","Padding","same")
    batchNormalizationLayer("Name","block_10_depthwise_BN","Epsilon",0.001)
    clippedReluLayer(6,"Name","block_10_depthwise_relu")
    convolution2dLayer([1 1],96,"Name","block_10_project","Padding","same")
    batchNormalizationLayer("Name","block_10_project_BN","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],576,"Name","block_11_expand","Padding","same")
    batchNormalizationLayer("Name","block_11_expand_BN","Epsilon",0.001)
    clippedReluLayer(6,"Name","block_11_expand_relu")
    groupedConvolution2dLayer([3 3],1,576,"Name","block_11_depthwise","Padding","same")
    batchNormalizationLayer("Name","block_11_depthwise_BN","Epsilon",0.001)
    clippedReluLayer(6,"Name","block_11_depthwise_relu")
    convolution2dLayer([1 1],96,"Name","block_11_project","Padding","same")
    batchNormalizationLayer("Name","block_11_project_BN","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","block_11_add");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],576,"Name","block_12_expand","Padding","same")
    batchNormalizationLayer("Name","block_12_expand_BN","Epsilon",0.001)
    clippedReluLayer(6,"Name","block_12_expand_relu")
    groupedConvolution2dLayer([3 3],1,576,"Name","block_12_depthwise","Padding","same")
    batchNormalizationLayer("Name","block_12_depthwise_BN","Epsilon",0.001)
    clippedReluLayer(6,"Name","block_12_depthwise_relu")
    convolution2dLayer([1 1],96,"Name","block_12_project","Padding","same")
    batchNormalizationLayer("Name","block_12_project_BN","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","block_12_add")
    convolution2dLayer([1 1],576,"Name","block_13_expand","Padding","same")
    batchNormalizationLayer("Name","block_13_expand_BN","Epsilon",0.001)
    clippedReluLayer(6,"Name","block_13_expand_relu")
    groupedConvolution2dLayer([3 3],1,576,"Name","block_13_depthwise","Padding","same","Stride",[2 2])
    batchNormalizationLayer("Name","block_13_depthwise_BN","Epsilon",0.001)
    clippedReluLayer(6,"Name","block_13_depthwise_relu")
    convolution2dLayer([1 1],160,"Name","block_13_project","Padding","same")
    batchNormalizationLayer("Name","block_13_project_BN","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],960,"Name","block_14_expand","Padding","same")
    batchNormalizationLayer("Name","block_14_expand_BN","Epsilon",0.001)
    clippedReluLayer(6,"Name","block_14_expand_relu")
    groupedConvolution2dLayer([3 3],1,960,"Name","block_14_depthwise","Padding","same")
    batchNormalizationLayer("Name","block_14_depthwise_BN","Epsilon",0.001)
    clippedReluLayer(6,"Name","block_14_depthwise_relu")
    convolution2dLayer([1 1],160,"Name","block_14_project","Padding","same")
    batchNormalizationLayer("Name","block_14_project_BN","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","block_14_add");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],960,"Name","block_15_expand","Padding","same")
    batchNormalizationLayer("Name","block_15_expand_BN","Epsilon",0.001)
    clippedReluLayer(6,"Name","block_15_expand_relu")
    groupedConvolution2dLayer([3 3],1,960,"Name","block_15_depthwise","Padding","same")
    batchNormalizationLayer("Name","block_15_depthwise_BN","Epsilon",0.001)
    clippedReluLayer(6,"Name","block_15_depthwise_relu")
    convolution2dLayer([1 1],160,"Name","block_15_project","Padding","same")
    batchNormalizationLayer("Name","block_15_project_BN","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","block_15_add")
    convolution2dLayer([1 1],960,"Name","block_16_expand","Padding","same")
    batchNormalizationLayer("Name","block_16_expand_BN","Epsilon",0.001)
    clippedReluLayer(6,"Name","block_16_expand_relu")
    groupedConvolution2dLayer([3 3],1,960,"Name","block_16_depthwise","Padding","same")
    batchNormalizationLayer("Name","block_16_depthwise_BN","Epsilon",0.001)
    clippedReluLayer(6,"Name","block_16_depthwise_relu")
    convolution2dLayer([1 1],320,"Name","block_16_project","Padding","same")
    batchNormalizationLayer("Name","block_16_project_BN","Epsilon",0.001)
    convolution2dLayer([1 1],1280,"Name","Conv_1")
    batchNormalizationLayer("Name","Conv_1_bn","Epsilon",0.001)
    clippedReluLayer(6,"Name","out_relu")
    globalAveragePooling2dLayer("Name","global_average_pooling2d_1")
    fullyConnectedLayer(5,"Name","fc")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];
lgraph = addLayers(lgraph,tempLayers);

% clean up helper variable
clear tempLayers;


lgraph = connectLayers(lgraph,"block_1_project_BN","block_2_expand");
lgraph = connectLayers(lgraph,"block_1_project_BN","block_2_add/in2");
lgraph = connectLayers(lgraph,"block_2_project_BN","block_2_add/in1");
lgraph = connectLayers(lgraph,"block_3_project_BN","block_4_expand");
lgraph = connectLayers(lgraph,"block_3_project_BN","block_4_add/in2");
lgraph = connectLayers(lgraph,"block_4_project_BN","block_4_add/in1");
lgraph = connectLayers(lgraph,"block_4_add","block_5_expand");
lgraph = connectLayers(lgraph,"block_4_add","block_5_add/in2");
lgraph = connectLayers(lgraph,"block_5_project_BN","block_5_add/in1");
lgraph = connectLayers(lgraph,"block_6_project_BN","block_7_expand");
lgraph = connectLayers(lgraph,"block_6_project_BN","block_7_add/in2");
lgraph = connectLayers(lgraph,"block_7_project_BN","block_7_add/in1");
lgraph = connectLayers(lgraph,"block_7_add","block_8_expand");
lgraph = connectLayers(lgraph,"block_7_add","block_8_add/in2");
lgraph = connectLayers(lgraph,"block_8_project_BN","block_8_add/in1");
lgraph = connectLayers(lgraph,"block_8_add","block_9_expand");
lgraph = connectLayers(lgraph,"block_8_add","block_9_add/in2");
lgraph = connectLayers(lgraph,"block_9_project_BN","block_9_add/in1");
lgraph = connectLayers(lgraph,"block_10_project_BN","block_11_expand");
lgraph = connectLayers(lgraph,"block_10_project_BN","block_11_add/in2");
lgraph = connectLayers(lgraph,"block_11_project_BN","block_11_add/in1");
lgraph = connectLayers(lgraph,"block_11_add","block_12_expand");
lgraph = connectLayers(lgraph,"block_11_add","block_12_add/in2");
lgraph = connectLayers(lgraph,"block_12_project_BN","block_12_add/in1");
lgraph = connectLayers(lgraph,"block_13_project_BN","block_14_expand");
lgraph = connectLayers(lgraph,"block_13_project_BN","block_14_add/in2");
lgraph = connectLayers(lgraph,"block_14_project_BN","block_14_add/in1");
lgraph = connectLayers(lgraph,"block_14_add","block_15_expand");
lgraph = connectLayers(lgraph,"block_14_add","block_15_add/in2");
lgraph = connectLayers(lgraph,"block_15_project_BN","block_15_add/in1");

%Image size of model input layer
inputSize = lgraph.Layers(1).InputSize
 
 
%augmentation 
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
    'Plots','training-progress', ...
    'ExecutionEnvironment', 'parallel');

%Train Model
net = trainNetwork(augimdsTrain,lgraph,options);
 
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
MobileNet_Model_1 = net;
save MobileNet_Model_1
 





