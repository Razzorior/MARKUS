using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MathNet.Numerics.Statistics;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;

public class SphereManager : MonoBehaviour
{
    public Material inputMaterial;
    public Material igMaterial;
    public GameObject IGSpheres;
    public void InitIGVisualization(double[,] input, double[,] ig)
    {
        int x_shape = input.GetLength(0);
        int y_shape = input.GetLength(1);

        var gradient_input = new Gradient();

        // Blend color from blue at 0% to white at 50% to red at 100%
        var colors = new GradientColorKey[2];
        colors[0] = new GradientColorKey(Color.white, 0.0f);
        colors[1] = new GradientColorKey(Color.black, 1.0f);

        // Keep alphas at 1 for all times
        var alphas = new GradientAlphaKey[2];
        alphas[0] = new GradientAlphaKey(1.0f, 0.0f);
        alphas[1] = new GradientAlphaKey(1.0f, 1f);

        gradient_input.SetKeys(colors, alphas);

        var gradient_ig = new Gradient();

        // Blend color from blue at 0% to white at 50% to red at 100%
        colors = new GradientColorKey[2];
        colors[0] = new GradientColorKey(Color.red, 0.0f);
        colors[1] = new GradientColorKey(Color.blue, 1.0f);

        // Keep alphas at 1 for all times
        alphas = new GradientAlphaKey[2];
        alphas[0] = new GradientAlphaKey(0.55f, 0.0f);
        alphas[1] = new GradientAlphaKey(0.55f, 1f);

        gradient_ig.SetKeys(colors, alphas);

        float min = float.MaxValue;
        for (int index = 0; index < x_shape; index++)
        {
            double[] row = GetRow(ig, index);
            float tmp = (float)row.Min();
            if (tmp < min) min = tmp;
        }

        float max = float.MaxValue;
        for (int index = 0; index < x_shape; index++)
        {
            double[] row = GetRow(ig, index);
            float tmp = (float)row.Max();
            if (tmp > min) min = tmp;
        }

        for (int i = 0; i < x_shape; i++)
        {
            for (int j = 0; j < y_shape; j++)
            {
                float x = ((j % y_shape) - (y_shape / 2)) * 0.1f;
                float y = ((i % y_shape) - (y_shape / 2)) * -0.1f + 2;
                float z = 0f;

                GameObject go = Instantiate(IGSpheres, this.transform);
                go.transform.localPosition = new Vector3(x,y,z);
                Transform child1 = go.transform.GetChild(0);
                Color color = gradient_input.Evaluate((float)input[i, j] / 255f);
                child1.GetComponent<Renderer>().material.color = color;

                float value = 0f;
                if (ig[i, j] >= 0)
                {
                    value = (float)(ig[i, j] / max);
                }
                else
                {
                    value = (float)(ig[i, j] / Mathf.Abs((float)min));
                }
                Debug.Log((value + 1f) / 2f);
                Color color2 = gradient_ig.Evaluate((value + 1f) /2f);
                child1.GetChild(0).GetComponent<Renderer>().material.color = color2;
            }

        }
    }

    public static double[] GetRow(double[,] matrix, int rowIndex)
    {
        int numCols = matrix.GetLength(1);
        double[] row = new double[numCols];

        for (int i = 0; i < numCols; i++)
        {
            row[i] = matrix[rowIndex, i];
        }

        return row;
    }
}
