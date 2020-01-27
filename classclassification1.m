
data = load('vehicleTrainingData.mat');
trainingData = data.vehicleTrainingData;

dataDir = fullfile(toolboxdir('vision'),'visiondata');
trainingData.imageFilename = fullfile(dataDir,trainingData.imageFilename);

net = load('yolov2VehicleDetector.mat');
Igraph = net.lgraph;

options = trainingOptions('sgdm','InitialLearnRate',...
                            0.001,'Verbose',true,'MiniBatchSize',...
                            16,'MaxEpochs',30,...
                            'Shuffle','every-epoch',...
                            'VerboseFrequency',30,...
                      'CheckpointPath',tempdir);
 [detector,info] = trainYOLOv2ObjectDetector(trainingData,Igraph,options);
 
 detector;
 
%  figure;
%  plot(info.TrainingLoss)
%  grid on
%  xlabel('Number of iterations')
%  ylabel('Training Loss for each Iteration')
 
% img = imread('Car7.jpg'); 
% img = imresize(img,[524 524]);
% [bboxes,scores] = detect(detector,img);
% 
%  if(~isempty(bboxes))
%     img = insertObjectAnnotation(img,'rectangle',bboxes,scores);
%     imshow(img);
%  end
camera = webcam;
img = camera.snapshot;
img = imresize(img,[524 524]);

[bbox,scrores]= detect(detector,img);

if(~isempty(bbox))
    img = insertObjectAnnotation(img,'rectangle',bbox,scrores);
    imshow(img)
end
 