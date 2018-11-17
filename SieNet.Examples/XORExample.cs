﻿using SiaNet.Common;
using SiaNet.Model;
using SiaNet.Model.Data;
using SiaNet.Model.Layers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Accord.Math;

namespace SiaNet.Examples
{
    internal class XORExample
    {
        private static DataFrameList trainData;

        private static Sequential model;
	    private static CompiledModel compiledModel;

        private static Shape featureShape;
        private static Shape labelShape;

        public static void LoadData()
        {
            featureShape = new Shape(2);
            labelShape = new Shape(1);
            DataFrame frameX = new DataFrame(featureShape);
            DataFrame frameY = new DataFrame(labelShape);
            /*
            //One approach of building dataset
            trainData.Add(new List<float>() { 0, 0 }, 0);
            trainData.Add(new List<float>() { 0, 1 }, 1);
            trainData.Add(new List<float>() { 1, 0 }, 1);
            trainData.Add(new List<float>() { 1, 1 }, 0);
            trainData.YFrame.OneHotEncode();
            */

            //Second approach
            frameX.Add(0, 0); frameY.Add(0);
            frameX.Add(0, 1); frameY.Add(1);
            frameX.Add(1, 0); frameY.Add(1);
            frameX.Add(1, 1); frameY.Add(0);

            trainData = new DataFrameList(frameX, frameY);
        }

        public static void BuildModel()
        {
            model = new Sequential(featureShape);
            model.Add(new Dense(dim: 2, weightInitializer: new Model.Initializers.Xavier()));
            model.Add(new Dense(dim: 1));
        }

        public static void Train()
        {
            //model.Compile(OptOptimizers.SGD, OptLosses.CrossEntropy, OptMetrics.Accuracy);
            compiledModel = model.Compile();
            compiledModel.EpochEnd += CompiledModel_EpochEnd;
            compiledModel.Fit(trainData, 100, 2, new Model.Optimizers.SGD(), new Model.Metrics.BinaryCrossEntropy(), new Model.Metrics.Accuracy());
        }

	    public static void Predict()
	    {
		    var predictions = compiledModel.Predict(trainData.Features);
		    for (int i = 0; i < trainData.Length; i++)
		    {
				Console.WriteLine($@"Data: {trainData.Features[i].ToOctave()}, Label: {trainData.Labels[i].ToOctave()}, Prediction: {predictions[i].ToOctave()}");
			}
			
	    }

		private static void CompiledModel_EpochEnd(object sender, EventArgs.EpochEndEventArgs e)
        {
            Console.WriteLine($@"Epoch: {e.Epoch}, Loss: {e.Loss}, Acc: {e.Metric}");
        }
    }
}
