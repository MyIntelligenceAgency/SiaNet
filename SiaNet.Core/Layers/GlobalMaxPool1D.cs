﻿using CNTK;

namespace SiaNet.Layers
{
    /// <summary>
    ///     Global max pooling operation for temporal data.
    /// </summary>
    /// <seealso cref="LayerBase" />
    public class GlobalMaxPool1D : LayerBase
    {
        /// <inheritdoc />
        internal override Function ToFunction(Variable inputFunction)
        {
            return CNTKLib.Pooling(inputFunction, PoolingType.Max, new[] {inputFunction.Shape[0]});
        }
    }
}