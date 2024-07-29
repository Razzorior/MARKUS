using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UMAP;
using System;

public class UmapReduction
{

    // TODO: Create Function that returns the positions of neuron IDS? Dictionary? the grid coordinates? The RAW data?
    public float[][] applyUMAP(float[][] weight_matrix, float max_size = float.MaxValue)
    {
        Umap umap = new Umap(distance: Umap.DistanceFunctions.Euclidean);
        int numberOfEpochs = umap.InitializeFit(weight_matrix);

        for (int i = 0; i < numberOfEpochs; i++)
        {
            umap.Step();
        }

        float[][] result = umap.GetEmbedding();

        /*
        float[][] result = Center_at_zero(umap.GetEmbedding());

        if (max_size != float.MaxValue)
        {
            result = Limit_size(result, max_size);
        }*/

        return result;
    }

    // This function assumes 2D embeddings. 
    public float[][] Center_at_zero(float[][] embeddings)
    {
        // Check if embeddings has exactly 2 dimensions
        if (embeddings[0].Length != 2)
        {
            throw new ArgumentException("Embeddings array must have exactly 2 dimensions.");
        }

        float[] mean = new float[2]; 
        for (int i = 0; i < embeddings.Length; i++)
        {
            mean[0] += embeddings[i][0];
            mean[1] += embeddings[i][1];
        }

        mean[0] /= embeddings.Length;
        mean[1] /= embeddings.Length;

        for(int i = 0; i < embeddings.Length; i++)
        {
            embeddings[i][0] -= mean[0]; 
            embeddings[i][1] -= mean[1];
        }

        return embeddings;
    }

    // If embedings are larger than the max_size on any side, all values get scaled down
    public float[][] Limit_size(float[][] embeddings, float max_size)
    {
        float max_value = 0f;

        // Allocate memory 
        float abs_x;
        float abs_y;

        foreach (float[] coordinate in embeddings)
        {
            abs_x = Mathf.Abs(coordinate[0]);
            abs_y = Mathf.Abs(coordinate[1]);
            if (abs_x > max_value) { max_value = abs_x; }
            if (abs_y > max_value) { max_value = abs_y; }
        }

        // If in bounds, return original embeddings
        if (max_value > max_size)
        {
            float scale_factor = max_size / max_value;

            foreach (float[] coordinate in embeddings)
            {
                coordinate[0] *= scale_factor;
                coordinate[1] *= scale_factor;
            }
        }

        return embeddings;
    }

        public float[][] CombineArrays(float[][] array1, float[][] array2)
    {
        int numRows1 = array1.Length;
        int numRows2 = array2.Length;
        int numCols1 = array1[0].Length;
        int numCols2 = array2[0].Length;

        // Check if dimensions match
        if (numCols1 != numRows2)
        {
            throw new ArgumentException("Number of columns in array1 must match number of rows in array2");
        }

        float[][] combinedArray = new float[numRows2][];

        for (int i = 0; i < numRows2; i++)
        {
            combinedArray[i] = new float[numRows1 + numCols2];

            // Copy elements from array1
            for (int j = 0; j < numRows1; j++)
            {
                combinedArray[i][j] = array1[j][i];
            }

            // Copy elements from array2
            for (int j = 0; j < numCols2; j++)
            {
                combinedArray[i][numCols1 + j] = array2[i][j];
            }
        }

        return combinedArray;
    }
}
