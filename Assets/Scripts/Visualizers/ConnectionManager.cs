using System.Collections;
using System.Linq;
using System;
using System.Collections.Generic;
using UnityEngine;

public class ConnectionManager : MonoBehaviour
{
    public Material line_mat;

    private List<LineRenderer> active_line_renderers = new List<LineRenderer>();
    private List<(int, int)> active_pos;
    private double[,] signals;
    List<ParticleSystem> _neuron_particles;
    Transform _tf1;
    Transform _tf2;

    /*
     * Obsolete at the moment. Might want to implement it using enum tasks instead of multiple functions
     * */
    public enum line_method
    {
        fix_number,
        all
    }

    private void Start()
    {
        // Making sure that a line material was set. 
        if (line_mat == null)
        {
            Debug.LogError("No Material Set for the Connection Manager!");
            Destroy(this.gameObject);
        }

    }

    private line_method used_method = line_method.fix_number;
    private int connectivity_amount = 100;

    /**
     * Method for most important connections between neurons of two layers, where only the
     * n-most (max_amount) active connections are drawn.
     */
    public void InitFixedConnectivity(double[,] weights, int max_amount, List<ParticleSystem> neuron_particles, Transform tf1, Transform tf2, (double? min, double? max)? scale = null, bool use_max_values = true)
    {
        if (weights.Length < max_amount)
        {
            Debug.LogError("Only " + weights.Length + " Connections possible, however " + max_amount + " where requested! (InitFixedConnectivity)");
            return;
        }
        connectivity_amount = max_amount;
        _neuron_particles = neuron_particles;
        _tf1 = tf1;
        _tf2 = tf2;
        signals = weights;

        var particles_layer_a = new ParticleSystem.Particle[weights.GetLength(0)];
        neuron_particles[0].GetParticles(particles_layer_a);

        var particles_layer_b = new ParticleSystem.Particle[weights.GetLength(1)];
        neuron_particles[1].GetParticles(particles_layer_b);

        List<(int, int)> n_positions = new List<(int, int)>();
        if (use_max_values)
        {
            n_positions = GetNLargestWeights(weights, max_amount);
        }
        else
        {
            n_positions = GetNSmallestWeights(weights, max_amount);
        }

        active_pos = n_positions;

        // Create Color Gradient [0, 1] 
        var color_gradient = new Gradient();

        // Blend color from blue at 0% to white at 50% to red at 100%
        var colors = new GradientColorKey[3];
        colors[0] = new GradientColorKey(Color.blue, 0.0f);
        colors[1] = new GradientColorKey(Color.white, 0.5f);
        colors[2] = new GradientColorKey(Color.red, 1.0f);

        // Keep alphas at 1 for all times
        var alphas = new GradientAlphaKey[3];
        alphas[0] = new GradientAlphaKey(1.0f, 0.0f);
        alphas[1] = new GradientAlphaKey(1.0f, 0.5f);
        alphas[2] = new GradientAlphaKey(1.0f, 1.0f);

        color_gradient.SetKeys(colors, alphas);

        // Get the dimensions of the array
        int rows = weights.GetLength(0);
        int cols = weights.GetLength(1);

        double min;
        double max;

        if (scale.HasValue)
        {
            min = (double)scale.Value.min;
            max = (double)scale.Value.max;
        }
        else
        {
            // Initialize min and max with the first element of the array
            min = weights[0, 0];
            max = weights[0, 0];

            // Iterate through the array to find min and max
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    double currentValue = weights[i, j];

                    // Update min and max values if necessary
                    if (currentValue < min)
                    {
                        min = currentValue;
                    }
                    else if (currentValue > max)
                    {
                        max = currentValue;
                    }
                }
            }
        }

        int line_number = 1;
        foreach ((int, int) position in n_positions)
        {
            // Create new Gameobject as child objekt 
            GameObject new_line = new GameObject("line_" + line_number.ToString());
            new_line.transform.parent = this.transform;
            // Add LineRenderer Component to display weight connectivty
            LineRenderer lr = new_line.AddComponent<LineRenderer>();
            // TODO: Shift values to extern public value, so that they can be changed from editor / other script
            lr.startWidth = 0.005f;
            lr.endWidth = 0.005f;

            // Set position of connection.
            lr.positionCount = 2;
            lr.useWorldSpace = true;
            lr.SetPosition(0, tf1.TransformPoint(particles_layer_a[position.Item1].position));
            lr.SetPosition(1, tf2.TransformPoint(particles_layer_b[position.Item2].position));

            float value = 0f;
            if (weights[position.Item1, position.Item2] >= 0)
            {
                value = (float)(weights[position.Item1, position.Item2] / (float)max);
            }
            else
            {
                value = (float)(weights[position.Item1, position.Item2] / Mathf.Abs((float)min));
            }

            Color line_color = color_gradient.Evaluate((value + 1) / 2f);
            lr.startColor = line_color;
            lr.endColor = line_color;


            lr.material = line_mat;

            active_line_renderers.Add(lr);

            line_number++;
        }

    }

    public List<int> InitBackwardsPassOfLayers(double[,] _signals, int max_lines, List<ParticleSystem> neuron_particles, Transform tf1, Transform tf2, List<int> back_layer_ids, (double? min, double? max)? scale = null)
    {
        // Hashset ignores duplicats. Needs to be transformed into a regular List to return 
        HashSet<int> new_ids = new HashSet<int>();

        connectivity_amount = max_lines;
        _neuron_particles = neuron_particles;
        _tf1 = tf1;
        _tf2 = tf2;
        signals = _signals;

        var particles_layer_a = new ParticleSystem.Particle[_signals.GetLength(0)];
        neuron_particles[0].GetParticles(particles_layer_a);

        var particles_layer_b = new ParticleSystem.Particle[_signals.GetLength(1)];
        neuron_particles[1].GetParticles(particles_layer_b);

        back_layer_ids.Sort();

        double[,] filtered_list = FilterArrayByList(_signals, back_layer_ids);

        // Since the list has been filtered, the second int means the position of the real index inside of the back_layer_ids.
        List<(int, int)> n_positions = GetNLargestWeights(filtered_list, max_lines);
        n_positions = GetRealPositions(n_positions, back_layer_ids);

        foreach ((int x, int y) in n_positions)
        {
            new_ids.Add(x);
        }

        active_pos = n_positions;

        // Create Color Gradient [0, 1] 
        var color_gradient = new Gradient();

        // Blend color from blue at 0% to white at 50% to red at 100%
        var colors = new GradientColorKey[3];
        colors[0] = new GradientColorKey(Color.blue, 0.0f);
        colors[1] = new GradientColorKey(Color.white, 0.5f);
        colors[2] = new GradientColorKey(Color.red, 1.0f);

        // Keep alphas at 1 for all times
        var alphas = new GradientAlphaKey[3];
        alphas[0] = new GradientAlphaKey(1.0f, 0.0f);
        alphas[1] = new GradientAlphaKey(1.0f, 0.5f);
        alphas[2] = new GradientAlphaKey(1.0f, 1.0f);

        color_gradient.SetKeys(colors, alphas);

        // Get the dimensions of the array
        int rows = filtered_list.GetLength(0);
        int cols = filtered_list.GetLength(1);

        double min;
        double max;

        if (scale.HasValue)
        {
            min = (double)scale.Value.min;
            max = (double)scale.Value.max;
        }
        else
        {
            // Initialize min and max with the first element of the array
            min = filtered_list[0, 0];
            max = filtered_list[0, 0];

            // Iterate through the array to find min and max
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    double currentValue = filtered_list[i, j];

                    // Update min and max values if necessary
                    if (currentValue < min)
                    {
                        min = currentValue;
                    }
                    else if (currentValue > max)
                    {
                        max = currentValue;
                    }
                }
            }
        }


        int line_number = 1;
        foreach ((int, int) position in n_positions)
        {
            // Create new Gameobject as child objekt 
            GameObject new_line = new GameObject("line_" + line_number.ToString());
            new_line.transform.parent = this.transform;
            // Add LineRenderer Component to display weight connectivty
            LineRenderer lr = new_line.AddComponent<LineRenderer>();
            // TODO: Shift values to extern public value, so that they can be changed from editor / other script
            lr.startWidth = 0.005f;
            lr.endWidth = 0.005f;

            // Set position of connection.
            lr.positionCount = 2;
            lr.useWorldSpace = true;
            lr.SetPosition(0, tf1.TransformPoint(particles_layer_a[position.Item1].position));
            lr.SetPosition(1, tf2.TransformPoint(particles_layer_b[position.Item2].position));

            float value = 0f;
            if (_signals[position.Item1, position.Item2] >= 0)
            {
                value = (float)(_signals[position.Item1, position.Item2] / (float)max);
            }
            else
            {
                value = (float)(_signals[position.Item1, position.Item2] / Mathf.Abs((float)min));
            }

            Color line_color = color_gradient.Evaluate((value + 1) / 2f);
            lr.startColor = line_color;
            lr.endColor = line_color;


            lr.material = line_mat;

            active_line_renderers.Add(lr);

            line_number++;
        }

        return new List<int>(new_ids);
    }

    public void DataManagerUpdate()
    {
        UpdateLinesForClustersInitFixedConnectivity();
    }

    public void UpdateLinesForClustersInitFixedConnectivity()
    {
        int index = 0;
        var particles_layer_a = new ParticleSystem.Particle[_neuron_particles[0].particleCount];
        _neuron_particles[0].GetParticles(particles_layer_a);

        var particles_layer_b = new ParticleSystem.Particle[_neuron_particles[1].particleCount];
        _neuron_particles[1].GetParticles(particles_layer_b);
        foreach ((int, int) position in active_pos)
        {
            active_line_renderers[index].SetPosition(0, _tf1.TransformPoint(particles_layer_a[position.Item1].position));
            active_line_renderers[index].SetPosition(1, _tf2.TransformPoint(particles_layer_b[position.Item2].position));
            index++;
        }
    }

    private List<(int, int)> GetNLargestWeights(double[,] weights, int max_amount)
    {

        int rows = weights.GetLength(0);
        int columns = weights.GetLength(1);

        var absolut_values_with_positions = new List<(double, int, int)>();
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                absolut_values_with_positions.Add((Math.Abs(weights[i, j]), i, j));
            }
        }

        //var sorted_positions = absolut_values_with_positions.OrderByDescending(v => v.Item1).Take(max_amount).ToList();
        var sorted_positions = absolut_values_with_positions.OrderByDescending(v => v.Item1).Take(max_amount).Select(p => (p.Item2, p.Item3)).ToList();

        return sorted_positions;
    }

    private List<(int, int)> GetNSmallestWeights(double[,] weights, int max_amount)
    {

        int rows = weights.GetLength(0);
        int columns = weights.GetLength(1);

        var absolut_values_with_positions = new List<(double, int, int)>();
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                absolut_values_with_positions.Add((Math.Abs(weights[i, j]), i, j));
            }
        }

        var sorted_positions = absolut_values_with_positions.OrderBy(v => v.Item1).Take(max_amount).Select(p => (p.Item2, p.Item3)).ToList();

        return sorted_positions;
    }

    private List<(int, int)> GetRealPositions(List<(int, int)> old_positions, List<int> actual_indicies)
    {
        List<(int, int)> result = new List<(int, int)>();


        foreach ((int x, int y) in old_positions)
        {
            result.Add((x, actual_indicies[y]));
        }

        return result;
    }

    double[,] FilterArrayByList(double[,] originalArray, List<int> indexList)
    {
        // Erstellen eines neuen Arrays mit der gleichen Anzahl von Zeilen und der Länge der indexList für die Spalten
        double[,] filteredArray = new double[originalArray.GetLength(0), indexList.Count];

        int filteredArray_j = 0;

        foreach (int wanted_index in indexList)
        {
            for (int i = 0; i < originalArray.GetLength(0); i++)
            {
                filteredArray[i, filteredArray_j] = originalArray[i, wanted_index];
            }
            filteredArray_j++;
        }
        
        return filteredArray;
    }
}
