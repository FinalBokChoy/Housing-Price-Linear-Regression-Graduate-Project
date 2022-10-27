numelstrain = 1000; % Number of elements in training data set
numelstest = 1000;  % Number of elements in testing data set

% -- TRAINING DATA SET -- %
trainx = rand(1, numelstrain)*5;
trainx = sort(trainx);
trainy = 2*trainx.^2 + 12*trainx + 23;

trainy = trainy + randn(1, numelstrain) * 5;   %Add gaussian noise

% -- TESTING DATA SET -- %
testx = rand(1, numelstest)*5;
testx = sort(testx);
testy = 2*testx.^2 + 12*testx + 23;

testy = testy + randn(1, numelstest) * 5;   %Add gaussian noise

%Polyfits
%1st order fit
trainp1 = polyfit(trainx, trainy, 1)
trainp2 = polyfit(trainx, trainy, 2)
trainp4 = polyfit(trainx, trainy, 4)

%Calculate output for training data
trainy1 = polyval(trainp1, trainx);
trainy2 = polyval(trainp2, trainx);
trainy4 = polyval(trainp4, trainx);

%Plot polyfits with training data
figure
plot(trainx,trainy,'o')
hold on
plot(trainx,trainy1)
plot(trainx,trainy2)
plot(trainx,trainy4)
hold off
title('Training data')

%Calculate error for training data
trainerr1 = 0;
for i = 1:numelstrain
    trainerr1 = trainerr1 + (log(trainy1(i)+1) - log(trainy(i)+1))^2;
end
trainerr1 = sqrt(trainerr1/numelstrain)

trainerr2 = 0;
for i = 1:numelstrain
    trainerr2 = trainerr2 + (log(trainy2(i)+1) - log(trainy(i)+1))^2;
end
trainerr2 = sqrt(trainerr2/numelstrain)

trainerr4 = 0;
for i = 1:numelstrain
    trainerr4 = trainerr4 + (log(trainy4(i)+1) - log(trainy(i)+1))^2;
end
trainerr4 = sqrt(trainerr4/numelstrain)

%Calculate output for testing data
testy1 = polyval(trainp1, testx);
testy2 = polyval(trainp2, testx);
testy4 = polyval(trainp4, testx);

%Plot polyfits with testing data
figure
plot(testx,testy,'o')
hold on
plot(testx,testy1)
plot(testx,testy2)
plot(testx,testy4)
hold off
title('Testing data')

%Calculate error for training data
testerr1 = 0;
for i = 1:numelstest
    testerr1 = testerr1 + (log(testy1(i)+1) - log(trainy(i)+1))^2;
end
testerr1 = sqrt(testerr1/numelstest)

testerr2 = 0;
for i = 1:numelstest
    testerr2 = testerr2 + (log(testy2(i)+1) - log(trainy(i)+1))^2;
end
testerr2 = sqrt(testerr2/numelstest)

testerr4 = 0;
for i = 1:numelstest
    testerr4 = testerr4 + (log(testy4(i)+1) - log(trainy(i)+1))^2;
end
testerr4 = sqrt(testerr4/numelstest)