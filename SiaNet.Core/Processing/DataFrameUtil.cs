﻿using CNTK;
using SiaNet;
using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SiaNet.Processing
{
    internal class DataFrameUtil
    {
        internal static Value GetValueBatch(DataFrame frame)
        {
            List<float> batch = new List<float>();
            frame.Data.ForEach((record) =>
            {
                batch.AddRange(record);
            });

            Value result = null;

            if (frame.FrameType == FrameType.IMG)
            {
                result = Value.CreateBatch(frame.Shape, batch, GlobalParameters.Device);
            }
            else if (frame.FrameType == FrameType.CSV)
            {
                result = Value.CreateBatch(new int[] { frame.Shape[1] }, batch, GlobalParameters.Device);
            }

            return result;
        }
    }
}
