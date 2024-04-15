using System.Collections;
using System.Linq;
using System;
using System.Collections.Generic;
using UnityEngine;

public class ConnectionManager : MonoBehaviour
{
    public Material line_mat;

    private List<LineRenderer> active_line_renderers = new List<LineRenderer>();
    private List<(int, int)> active_largest_pos;
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
     * n-most (max_amoiunt) active connections are drawn.
     */
    public void InitFixedConnectivity(double[,] weights, int max_amount, List<ParticleSystem> neuron_particles, Transform tf1, Transform tf2)
    {
        // TODO: Check if there this is out of bounds (max amount larger than number of activations)
        connectivity_amount = max_amount;
        _neuron_particles = neuron_particles;
        _tf1 = tf1;
        _tf2 = tf2;

        var particles_layer_a = new ParticleSystem.Particle[weights.GetLength(0)];
        neuron_particles[0].GetParticles(particles_layer_a);

        var particles_layer_b = new ParticleSystem.Particle[weights.GetLength(1)];
        neuron_particles[1].GetParticles(particles_layer_b);

        List<(int, int)> largest_n_positions = GetNLargestWeights(weights, max_amount);
        active_largest_pos = largest_n_positions;

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

        // Initialize min and max with the first element of the array
        double min = weights[0, 0];
        double max = weights[0, 0];

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

        int line_number = 1;
        foreach ((int, int) position in largest_n_positions)
        {
            // Create new Gameobject as child objekt 
            GameObject new_line = new GameObject("line_" + line_number.ToString());
            new_line.transform.parent = this.transform;
            // Instantiate(new_line);
            // Add LineRenderer Component to display weight connectivty
            LineRenderer lr = new_line.AddComponent<LineRenderer>();
            // TODO: Shift values to extern public value, so that they can be changed from editor / other script
            lr.startWidth = 0.005f;
            lr.endWidth = 0.005f;

            // Set position of connection.
            // TODO: Check whether these values are global coordinates per default, or if this is a nother setting that needs to be considered.
            lr.positionCount = 2;
            lr.useWorldSpace = true;
            lr.SetPosition(0, tf1.TransformPoint(particles_layer_a[position.Item1].position));
            lr.SetPosition(1, tf2.TransformPoint(particles_layer_b[position.Item2].position));

            float value = 0f;
            if (weights[position.Item1, position.Item2] >= 0)
            {
                value = (float)(weights[position.Item1, position.Item2] / max);
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
        foreach ((int, int) position in active_largest_pos)
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

}
