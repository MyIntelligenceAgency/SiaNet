﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using SiaNet;
using SiaNet.Layers;
using CNTK;
using Samples.Common;
using SiaNet.Common;

namespace Samples.GPU
{
    class Program
    {
        static void Main(string[] args)
        {
            try
            {
                var devices = DeviceDescriptor.AllDevices().Where(x=>(x.Type == DeviceKind.GPU)).ToList();
                if (devices.Count == 0)
                    throw new Exception("No GPU Device found. Please run the CPU examples instead!");

                //Logging.OnWriteLog += Logging_OnWriteLog;

                //Setting global device
                GlobalParameters.Device = devices[0];
                
                //XOR Example
                XORExample.LoadData();
                XORExample.BuildModel();
                XORExample.Train();
                
                ////Housing regression example
                //HousingRegression.LoadData();
                //HousingRegression.BuildModel();
                //HousingRegression.Train();

                //MNIST Classification example
                //MNISTClassifier.LoadData();
                //MNISTClassifier.BuildModel();
                //MNISTClassifier.Train();

                ////Time series prediction
                //TimeSeriesPrediction.LoadData();
                //TimeSeriesPrediction.BuildModel();
                //TimeSeriesPrediction.Train();
                

                ////Multi variate time series prediction
                //MiltiVariateTimeSeriesPrediction.LoadData();
                //MiltiVariateTimeSeriesPrediction.BuildModel();
                //MiltiVariateTimeSeriesPrediction.Train();

                ////Cifar-10 Classification example
                //Cifar10Classification.LoadData();
                //Cifar10Classification.BuildModel();
                //Cifar10Classification.Train();

                ////Image classification example
                //Console.WriteLine("ResNet50 Prediction: " + ImageClassification.ImagenetTest(Common.ImageNetModel.ResNet50)[0].Name);
                //Console.WriteLine("Cifar 10 Prediction: " + ImageClassification.Cifar10Test(Common.Cifar10Model.ResNet110)[0].Name);


                //Object Detection
                //ObjectDetection.PascalDetection();
                //ObjectDetection.GroceryDetection();
                Console.ReadLine();
            }
            catch(Exception ex)
            {
                Console.WriteLine(ex.ToString());
                Console.ReadLine();
            }
        }

        private static void Logging_OnWriteLog(string message)
        {
            Console.WriteLine("Log Message: " + message);
        }
    }
}
