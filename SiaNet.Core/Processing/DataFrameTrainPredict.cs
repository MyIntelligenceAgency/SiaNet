﻿using CNTK;
using SiaNet.Events;
using SiaNet.Interface;
using SiaNet;
using System.Collections.Generic;
using System.Linq;

namespace SiaNet.Processing
{
    internal class DataFrameTrainPredict : ITrainPredict
    {
        public Function Model;
        private Function lossFunc;
        private Function metricFunc;
        private IList<Learner> learners;
        private SiaVariable featureVariable;
        private SiaVariable labelVariable;
        private string lossName;
        private string metricName;

        public DataFrameTrainPredict(Function model, Function lossFunc, string lossName, Function metricFunc, string metricName, IList<Learner> learners, SiaVariable featureVariable, SiaVariable labelVariable)
        {
            this.Model = model;
            this.lossFunc = lossFunc;
            this.metricFunc = metricFunc;
            this.learners = learners;
            this.featureVariable = featureVariable;
            this.labelVariable = labelVariable;
            this.metricName = metricName;
            this.lossName = lossName;
        }

        public Dictionary<string, List<double>> Train(object trainData, object validationData, int epoches, int batchSize, On_Epoch_Start OnEpochStart, On_Epoch_End OnEpochEnd, On_Batch_Start onBatchStart, On_Batch_End OnBatchEnd, bool shuffle = false)
        {
            XYFrame train = (XYFrame)trainData;
            XYFrame validation = validationData != null ? (XYFrame)validationData : null;
            Dictionary<string, List<double>> result = new Dictionary<string, List<double>>();
            var trainer = Trainer.CreateTrainer(Model, lossFunc, metricFunc, learners);
            int currentEpoch = 1;
            Dictionary<string, double> metricsList = new Dictionary<string, double>();
            while (currentEpoch <= epoches)
            {
                if (shuffle)
                    train.Shuffle();

                metricsList = new Dictionary<string, double>();
                OnEpochStart(currentEpoch);
                int miniBatchCount = 1;
                while (train.NextBatch(miniBatchCount, batchSize))
                {
                    onBatchStart(currentEpoch, miniBatchCount);
                    Value features = DataFrameUtil.GetValueBatch(train.CurrentBatch.XFrame);
                    Value labels = DataFrameUtil.GetValueBatch(train.CurrentBatch.YFrame);

                    trainer.TrainMinibatch(new Dictionary<SiaVariable, Value>() { { featureVariable, features }, { labelVariable, labels } }, GlobalParameters.Device);
                    OnBatchEnd(currentEpoch, miniBatchCount, trainer.TotalNumberOfSamplesSeen(), trainer.PreviousMinibatchLossAverage(), new Dictionary<string, double>() { { metricName, trainer.PreviousMinibatchEvaluationAverage() } });
                    miniBatchCount++;
                }

                if (!result.ContainsKey("loss"))
                {
                    result.Add("loss", new List<double>());
                }

                if (!result.ContainsKey(metricName))
                {
                    result.Add(metricName, new List<double>());
                }

                double lossValue = trainer.PreviousMinibatchLossAverage();
                double metricValue = trainer.PreviousMinibatchEvaluationAverage();
                result["loss"].Add(lossValue);
                result[metricName].Add(metricValue);
                metricsList.Add(metricName, metricValue);
                if (validation != null)
                {
                    if (!result.ContainsKey("val_loss"))
                    {
                        result.Add("val_loss", new List<double>());
                    }

                    if (!result.ContainsKey("val_" + metricName))
                    {
                        result.Add("val_" + metricName, new List<double>());
                    }

                    int evalMiniBatchCount = 1;
                    List<double> totalEvalBatchLossList = new List<double>();
                    List<double> totalEvalMetricValueList = new List<double>();
                    while (validation.NextBatch(evalMiniBatchCount, batchSize))
                    {
                        SiaVariable actualVariable = CNTKLib.InputVariable(labelVariable.Shape, DataType.Float);
                        var evalLossFunc = Losses.Get(lossName, labelVariable, actualVariable);
                        var evalMetricFunc = Metrics.Get(metricName, labelVariable, actualVariable);
                        Value actual = EvaluateInternal(validation.CurrentBatch.XFrame);
                        Value expected = DataFrameUtil.GetValueBatch(validation.CurrentBatch.YFrame);
                        var inputDataMap = new Dictionary<SiaVariable, Value>() { { labelVariable, expected }, { actualVariable, actual } };
                        var outputDataMap = new Dictionary<SiaVariable, Value>() { { evalLossFunc.Output, null } };

                        evalLossFunc.Evaluate(inputDataMap, outputDataMap, GlobalParameters.Device);
                        var evalLoss = outputDataMap[evalLossFunc.Output].GetDenseData<float>(evalLossFunc.Output).Select(x => x.First()).ToList();
                        totalEvalBatchLossList.Add(evalLoss.Average());

                        inputDataMap = new Dictionary<SiaVariable, Value>() { { labelVariable, expected }, { actualVariable, actual } };
                        outputDataMap = new Dictionary<SiaVariable, Value>() { { evalMetricFunc.Output, null } };
                        evalMetricFunc.Evaluate(inputDataMap, outputDataMap, GlobalParameters.Device);
                        var evalMetric = outputDataMap[evalMetricFunc.Output].GetDenseData<float>(evalMetricFunc.Output).Select(x => x.First()).ToList();
                        totalEvalMetricValueList.Add(evalMetric.Average());

                        evalMiniBatchCount++;
                    }

                    result["val_loss"].Add(totalEvalBatchLossList.Average());
                    metricsList.Add("val_loss", totalEvalBatchLossList.Average());
                    result["val_" + metricName].Add(totalEvalMetricValueList.Average());
                    metricsList.Add("val_" + metricName, totalEvalMetricValueList.Average());
                }

                OnEpochEnd(currentEpoch, trainer.TotalNumberOfSamplesSeen(), lossValue, metricsList);
                currentEpoch++;
            }

            return result;
        }

        private Value EvaluateInternal(DataFrame data)
        {
            Value features = DataFrameUtil.GetValueBatch(data);
            var inputDataMap = new Dictionary<SiaVariable, Value>() { { featureVariable, features } };
            var outputDataMap = new Dictionary<SiaVariable, Value>() { { Model.Output, null } };
            Model.Evaluate(inputDataMap, outputDataMap, GlobalParameters.Device);
            return outputDataMap[Model.Output];
        }

        public IList<float> Evaluate(DataFrame data)
        {
            var outputValue = EvaluateInternal(data);
            IList<IList<float>> resultSet = outputValue.GetDenseData<float>(Model.Output);
            var result = resultSet[0];
            return result;
        }

        /// <inheritdoc />
        public void Dispose()
        {
            Model?.Dispose();
            lossFunc?.Dispose();
            metricFunc?.Dispose();
            featureVariable?.Dispose();
            labelVariable?.Dispose();
        }
    }
}
