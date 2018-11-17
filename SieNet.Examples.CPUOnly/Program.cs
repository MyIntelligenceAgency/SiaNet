﻿using CNTK;
using SiaNet;
using SiaNet.Application;
using SiaNet.Common;
using SiaNet.Model;
using SiaNet.Examples;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using SiaNet.Model.Layers;

namespace SieNet.Examples.CPUOnly
{
    class Program
    {
        static void Main(string[] args)
        {
            try
            {
                //Setting global device
                Logging.OnWriteLog += Logging_OnWriteLog;


				//XOR Example

				//XORExample.LoadData();
				//XORExample.BuildModel();
				//XORExample.Train();
				//         XORExample.Predict();



				//Housing regression example
				HousingRegression.LoadData();
				HousingRegression.BuildModel();
				HousingRegression.Train();

				//MNIST Classification example

				//MNISTClassifier.LoadData();
				//MNISTClassifier.BuildModel(true);
				//MNISTClassifier.Train();

				//LSTM Time series example
				//TimeSeriesPrediction.LoadData();
				//TimeSeriesPrediction.BuildModel();
				//TimeSeriesPrediction.Train();

				//Multi variate time series prediction

				//MiltiVariateTimeSeriesPrediction.LoadData();
				//MiltiVariateTimeSeriesPrediction.BuildModel();
				//MiltiVariateTimeSeriesPrediction.Train();

				//Cifar - 10 Classification example
				//Cifar10Classification.LoadData();
				//Cifar10Classification.BuildModel();
				//Cifar10Classification.Train();

				//Image classification example
				//Console.WriteLine("ResNet50 Prediction: " + ImageClassification.ImagenetTest(SiaNet.Common.ImageNetModel.ResNet50)[0].Name);
				//Console.WriteLine("Cifar 10 Prediction: " + ImageClassification.Cifar10Test(SiaNet.Common.Cifar10Model.ResNet110)[0].Name);

				//Object Detection
				//ObjectDetection.PascalDetection();
				//ObjectDetection.GroceryDetection();
				Console.ReadLine();
                
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
                Console.ReadLine();
            }
        }

        private void EvalImageNet()
        {

        }

        private static void Logging_OnWriteLog(string message)
        {
            Console.WriteLine(message);
        }
    }
}
