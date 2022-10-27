clear all

numelstrain = 1000; % Number of elements in training data set
numelstest = 460;  % Number of elements in testing data set

% Import data
%rawds = fileDatastore(fullfile(pwd,'ames iowa housing.csv'),'ReadFcn',@load,'FileExtensions','.csv');
filename = 'ames iowa housing.csv';
A = xlsread(filename);

% Read input data (Columns for lot area, year build, 1st floor, 2nd floor,
% garage areas
% Training data
trainx0 = ones(numelstrain, 1); % X0 vector with all 1s
trainx1 = A(1:numelstrain,5);
trainx2 = A(1:numelstrain,20);
trainx3 = A(1:numelstrain,44);
trainx4 = A(1:numelstrain,45);
trainx5 = A(1:numelstrain,63);

% Testing data
testx0 = ones(numelstest, 1);   % X0 vector with all 1s
testx1 = A(numelstrain + 1 : numelstrain + numelstest,5);
testx2 = A(numelstrain + 1 : numelstrain + numelstest,20);
testx3 = A(numelstrain + 1 : numelstrain + numelstest,44);
testx4 = A(numelstrain + 1 : numelstrain + numelstest,45);
testx5 = A(numelstrain + 1 : numelstrain + numelstest,63);

% Training data true results
trainy = A(1:numelstrain,81);

% Testing data true results
testy = A(numelstrain + 1 : numelstrain + numelstest,81);

% -- FIRST MODEL -- %
% Create input matrix (training data)
trainx(:,1) = trainx0;
trainx(:,2) = trainx1;
trainx(:,3) = trainx2;
trainx(:,4) = trainx3;
trainx(:,5) = trainx4;
trainx(:,6) = trainx5;
% Train model using linear regression fit
trainmdl = fitlm(trainx, trainy);
% Extract coeffs from model
coeffs = table2array(trainmdl.Coefficients(2:7, 1));
bias = table2array(trainmdl.Coefficients(1,1));
% Calculate y (predicted)
trainy1 = trainx*coeffs + bias;
% Calculate error
trainerr1 = 0;
for i = 1:numelstrain
    trainerr1 = trainerr1 + (log(trainy1(i)+1) - log(trainy(i)+1))^2;
end
trainerr1 = sqrt(trainerr1/numelstrain)

% Create input matrix (test data)
testx(:,1) = testx0;
testx(:,2) = testx1;
testx(:,3) = testx2;
testx(:,4) = testx3;
testx(:,5) = testx4;
testx(:,6) = testx5;
% Calculate y (predicted)
testy1 = testx*coeffs + bias;
testerr1 = 0;
% Calculate error
for i = 1:numelstest
    testerr1 = testerr1 + (log(testy1(i)+1) - log(testy(i)+1))^2;
end
testerr1 = sqrt(testerr1/numelstest)

% Plot training data ('X' value = ID of house)
itrain = linspace(1, numelstrain, numelstrain);
tiledlayout(3,2)
nexttile
plot(itrain,trainy,'o')
hold on
plot(itrain,trainy1)
hold off
legend('True Price (USD)', 'Predicted Price (USD)')
title('Training data (Model 1)')
nexttile

% Plot testing data
itest = linspace(numelstrain+1, numelstrain+numelstest, numelstest);
%figure
plot(itest,testy,'o')
hold on
plot(itest,testy1)
hold off
legend('True Price (USD)', 'Predicted Price (USD)')
title('Testing data (Model 1)')
nexttile

% -- SECOND MODEL -- %
% Create input matrix (training data)
trainarr = [trainx1, trainx2, trainx3, trainx4, trainx5];
i = 1;
trainx(:,i) = trainx0;
for n = 1:size(trainarr,2)
    i = i+1;
    trainx(:,i) = trainarr(:,n);
end
for m = 1:size(trainarr,2)
    for n = 1:size(trainarr,2)
        i = i+1;
        trainx(:,i) = trainarr(:,m).*trainarr(:,n);
    end
end
    
% Train model using linear regression fit
trainmdl = fitlm(trainx, trainy);
% Extract coeffs from model
coeffs = table2array(trainmdl.Coefficients(2:32, 1));
bias = table2array(trainmdl.Coefficients(1,1));
% Calculate y (predicted)
trainy2 = trainx*coeffs + bias;
% Calculate error
trainerr2 = 0;
for i = 1:numelstrain
    trainerr2 = trainerr2 + (log(trainy2(i)+1) - log(trainy(i)+1))^2;
end
trainerr2 = sqrt(trainerr2/numelstrain)

% Create input matrix (testing data)
testarr = [testx1, testx2, testx3, testx4, testx5];
i = 1;
testx(:,i) = testx0;
for n = 1:size(testarr,2)
    i = i+1;
    testx(:,i) = testarr(:,n);
end
for m = 1:size(testarr,2)
    for n = 1:size(testarr,2)
        i = i+1;
        testx(:,i) = testarr(:,m).*testarr(:,n);
    end
end
% Calculate y (predicted)
testy2 = testx*coeffs + bias;
% Calculate error
testerr2 = 0;
for i = 1:numelstest
    testerr2 = testerr2 + (log(testy2(i)+1) - log(testy(i)+1))^2;
end
testerr2 = sqrt(testerr2/numelstest)

% Plot training data
itrain = linspace(1, numelstrain, numelstrain);
%figure
plot(itrain,trainy,'o')
hold on
plot(itrain,trainy2)
hold off
legend('True Price (USD)', 'Predicted Price (USD)')
title('Training data (Model 2)')
nexttile

% Plot testing data
itest = linspace(numelstrain+1, numelstrain+numelstest, numelstest);
%figure
plot(itest,testy,'o')
hold on
plot(itest,testy2)
hold off
legend('True Price (USD)', 'Predicted Price (USD)')
title('Testing data (Model 2)')
nexttile

% -- THIRD MODEL -- %
% Create input matrix (training data)
i = 1;
trainx(:,i) = trainx0;
for n = 1:size(trainarr,2)
    i = i+1;
    trainx(:,i) = trainarr(:,n);
end
for m = 1:size(trainarr,2)
    for n = 1:size(trainarr,2)
        for o = 1:size(trainarr,2)
            for p = 1:size(trainarr,2)
                i = i+1;
                trainx(:,i) = trainarr(:,m).*trainarr(:,n).*trainarr(:,o).*trainarr(:,p);
            end
        end
    end
end
    
% Train model using linear regression fit
trainmdl = fitlm(trainx, trainy);
% Extract coeffs from model
coeffs = table2array(trainmdl.Coefficients(2:632, 1));
bias = table2array(trainmdl.Coefficients(1,1));
% Calculate y (predicted)
trainy3 = trainx*coeffs + bias;
% Calculate error
trainerr3 = 0;
for i = 1:numelstrain
    trainerr3 = trainerr3 + (log(trainy3(i)+1) - log(trainy(i)+1))^2;
end
trainerr3 = sqrt(trainerr3/numelstrain)

% Create input matrix (testing data)
i = 1;
testx(:,i) = testx0;
for n = 1:size(testarr,2)
    i = i+1;
    testx(:,i) = testarr(:,n);
end
for m = 1:size(testarr,2)
    for n = 1:size(testarr,2)
        for o = 1:size(testarr,2)
            for p = 1:size(testarr,2)
                i = i+1;
                testx(:,i) = testarr(:,m).*testarr(:,n).*testarr(:,o).*testarr(:,p);
            end
        end
    end
end
% Calculate y (predicted)
testy3 = testx*coeffs + bias;
% Calculate error
testerr3 = 0;
for i = 1:numelstest
    testerr3 = testerr3 + (log(testy3(i)+1) - log(testy(i)+1))^2;
end
testerr3 = sqrt(testerr3/numelstest)

% Plot training data
itrain = linspace(1, numelstrain, numelstrain);
%figure
plot(itrain,trainy,'o')
hold on
plot(itrain,trainy3)
hold off
legend('True Price (USD)', 'Predicted Price (USD)')
title('Training data (Model 3)')
nexttile

% Plot testing data
itest = linspace(numelstrain+1, numelstrain+numelstest, numelstest);
%figure
plot(itest,testy,'o')
hold on
plot(itest,testy3)
hold off
legend('True Price (USD)', 'Predicted Price (USD)')
title('Testing data (Model 3)')