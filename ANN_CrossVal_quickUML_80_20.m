clc;
clear;
M = csvread('dataset_quickUML_80_20.csv');
traindata1 = M(:,1:268);
trainlabel1 = M(:,269:269);
% Divide data 80% - Train data (traindata2,trainlabel2), 20%-Test data
% (testdata2,testlabel2)
traindata2 = traindata1(1:75,:);
trainlabel2 = trainlabel1(1:75,:);
testdata2 = traindata1(76:94,:);
testlabel2 = trainlabel1(76:94,:);
input  = traindata2;
target = trainlabel2;

%Division of data
% p=0.5;
% [Train , Test] = crossvalind('HoldOut', target, p);
% TrainSample = input(Train,:);
% TrainLabel = target(Train,1);
% TestSample = input(Test,:);
% TestLabel = target(Test,1);

%Learning/Training of ANN (5 Fold CV)
numFolds = 5;
accFold = 0;
Indices = crossvalind('Kfold', target, numFolds);
maxnodeval = -999;
acccheck = -1;
nodevalues = [5;10;15];


for i=1:numFolds
    TestingFoldSample = input(Indices==i,:);
    TrainingFoldSample = input(Indices~=i,:);
    TrainingFoldLabel = target(Indices~=i,:);
    TestingFoldLabel = target(Indices==i,:);
    for j = 1:size(nodevalues,1)
        nodeval = nodevalues(j);
        %disp(nodeval);
        net =newff(TrainingFoldSample',TrainingFoldLabel',nodeval);
        net.trainParam.epochs = 20;
        net = train(net,TrainingFoldSample',TrainingFoldLabel');
        output = net(TestingFoldSample');
        output=round(output);
        output(output<1)=1;
        output(output>4)=4;
        confval = confusionmat(TestingFoldLabel,output');
        fprintf('*** Fold %d Confusion Matrix ***\n',i);
        disp(confval);
        correctpred = 0;
        for index=1:size(confval,1);
            correctpred=correctpred+confval(index,index);
        end
        accuracy1 = correctpred/length(TestingFoldLabel);
            if (accuracy1>=acccheck)
                acccheck = accuracy1;
                maxnodeval = nodeval;
                correctpred1 = correctpred;
            end
    end
    accFold = accFold + correctpred1;
end
trainaccuracy = accFold/length(target);
fprintf('Train Accuracy =  %f\n',trainaccuracy*100);
fprintf('Best Hidden node value =  %d\n',maxnodeval);

%Testing ANN
net1 = newff(traindata2',trainlabel2',maxnodeval);
net1.trainParam.epochs = 20;
net1 = train(net1,traindata2',trainlabel2');
output1 = net1(testdata2');
output1=round(output1);
output1(output1<1)=1;
output1(output1>4)=4;
confval1 = confusionmat(testlabel2,output1');
 correctpred2 = 0;
        for index=1:size(confval1,1);
            correctpred2=correctpred2+confval1(index,index);
        end
testaccuracy = correctpred2/length(testlabel2);
fprintf('\n*** Confusion Matrix ***\n');
disp(confval1);
fprintf('Test Accuracy =  %f\n',testaccuracy*100);

classes = 4;
samples = 28;
label = testlabel2;
output = output1';
%Create label vector
labelval = zeros(samples,classes);
[labelrows, labelcols] = size(label);
for i = 1:labelrows
    index = label(i);
    %disp(index);
    labelval(i,index) = 1;
end
%Create output vector
outputval = zeros(samples,classes);
[outputrows, outputcols] = size(output);
for i = 1:outputrows
    index = output(i);
    %disp(index);
    outputval(i,index) = 1;
end
plotconfusion(labelval',outputval');