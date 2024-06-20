using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MathNet.Numerics.Statistics;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using System;

//Reload!
public class ParticleManager : MonoBehaviour
{
    [HideInInspector]
    public ParticleSystem ps;

    private float clustering_speed = 1.5f;
    private List<ClusteringRequest> clustering_Requests = new List<ClusteringRequest>();

    // Start is called before the first frame update
    void Start()
    {
        ps = this.GetComponent<ParticleSystem>();

        if (ps is null) 
        {
            Debug.Log("ParticleManager failed loading the particle System");
            return;
        }
    }

    // Update Loop Checks for Particle Movements requested ( For Clustering e.g. ) 
    public bool DataManagerUpdate()
    {
        // No movements to be done
        if (clustering_Requests.Count <= 0)
        {
            return false;
        }

        int particle_count = ps.particleCount;
        var particles = new ParticleSystem.Particle[particle_count];
        ps.GetParticles(particles);
        float current_time = Time.time;
        float t;

        // TODO: Maybe use ParticleSystems velocity instead, so that the particle list doesn't need to be edited?
        List<int> elementsToRemove = new List<int>();
        for (int index = 0; index < clustering_Requests.Count; index++)
        {
            t = (current_time - clustering_Requests[index].start_time) * clustering_speed;
            particles[clustering_Requests[index].id].position = Vector3.Lerp(clustering_Requests[index].start_position, clustering_Requests[index].target_position, t);
            // If t >= 1, the end position is reached, so the request can be removed
            if (t >= 1)
            {
                elementsToRemove.Add(index);
            }
        }

        // Clean up finished Requests
        elementsToRemove.Sort((a, b) => b.CompareTo(a));
        foreach (int index in elementsToRemove)
        {
            clustering_Requests.RemoveAt(index);
        }

        ps.SetParticles(particles);

        return true;
    }

    public void InitParticleSystems(int count)
    {
        ps.Play();
        ps.Emit(count);
        Shift_blank_particles(count);
    }

    public void InitParticleSystemsInput(double[,] input, int count)
    {
        ps.Play();
        ps.Emit(count);
        Shift_input_particles(input, count);
    }

    public void InitParticleSystemsNAPs(double[,] naps, int class_index)
    {
        ps.Play();
        int nap_count = naps.GetLength(1);
        ps.Emit(nap_count);

        Shift_nap_particles(naps, class_index);
    }

    public void InitParticleSystemsWB(double[] biases, bool coordinates_given = false, float[][] coodinates = null)
    {
        ps.Play();
        int biases_count = biases.GetLength(0);
        ps.Emit(biases_count);

        Shift_WB_particles(biases);
    }

    public void InitParticleSystemsWithGivenPositions(float[][] coordinates)
    {
        ps.Play();
        int neuron_count = coordinates.Length;
        ps.Emit(neuron_count);

        Shift_blank_particles_with_given_position(coordinates, neuron_count);
    }

    public void UpdateParticleSystems(double[,] naps, int class_index)
    {
        Shift_nap_particles(naps, class_index);
    }

    public void UpdateParticleSystemsInput(double[,] input)
    {
        int count = input.GetLength(0) * input.GetLength(1);
        var particles = new ParticleSystem.Particle[count];
        ps.GetParticles(particles);

        var gradient = new Gradient();

        // Blend color from blue at 0% to white at 50% to red at 100%
        var colors = new GradientColorKey[2];
        colors[0] = new GradientColorKey(Color.white, 0.0f);
        colors[1] = new GradientColorKey(Color.black, 1.0f);

        // Keep alphas at 1 for all times
        var alphas = new GradientAlphaKey[2];
        alphas[0] = new GradientAlphaKey(1.0f, 0.0f);
        alphas[1] = new GradientAlphaKey(1.0f, 1f);

        gradient.SetKeys(colors, alphas);

        int amount_per_row = Mathf.FloorToInt(Mathf.Sqrt(count));

        for (int index = 0; index < count; index++)
        {
            float input_value = (float)input[(int)Mathf.Floor(index / amount_per_row), index % amount_per_row];
            Color color_to_set = gradient.Evaluate(input_value / 255f);
            particles[index].startColor = color_to_set;
        }

        ps.SetParticles(particles, count);

    }

    public void UpdateParticleSytemsWB(double[] biases)
    {
        int count = biases.GetLength(0);
        var particles = new ParticleSystem.Particle[count];
        ps.GetParticles(particles);

        int amount_per_row = Mathf.FloorToInt(Mathf.Sqrt(count));
        double max = biases.Max();
        double min = biases.Min();

        var gradient = new Gradient();

        // Blend color from blue at 0 % to white at 50 % to red at 100 %
        var colors = new GradientColorKey[3];
        colors[0] = new GradientColorKey(Color.blue, 0.0f);
        colors[1] = new GradientColorKey(Color.white, 0.5f);
        colors[2] = new GradientColorKey(Color.red, 1.0f);

        // Keep alphas at 1 for all times
        var alphas = new GradientAlphaKey[3];
        alphas[0] = new GradientAlphaKey(1.0f, 0.0f);
        alphas[1] = new GradientAlphaKey(1.0f, 0.5f);
        alphas[2] = new GradientAlphaKey(1.0f, 1.0f);

        gradient.SetKeys(colors, alphas);

        for (int index = 0; index < count; index++)
        {
            float value = 0f;
            if (biases[index] >= 0)
            {
                value = (float)(biases[index] / max);
            }
            else
            {
                value = (float)(biases[index] / Mathf.Abs((float)min));
            }
            particles[index].startColor = gradient.Evaluate((value + 1) / 2f);
        }

        ps.SetParticles(particles, count);
    }

    private void Shift_blank_particles(int count)
    {
        var particles = new ParticleSystem.Particle[count];
        ps.GetParticles(particles);

        int amount_per_row = Mathf.FloorToInt(Mathf.Sqrt(count));

        for (int index = 0; index < count; index++)
        {
            particles[index].angularVelocity = 0;
            particles[index].angularVelocity3D = new Vector3(0, 0, 0);
            particles[index].velocity = new Vector3(0, 0, 0);
            particles[index].startColor = Color.grey;


            float x = ((index % amount_per_row) - (amount_per_row / 2)) * 0.1f;
            float y = (Mathf.Floor(index / amount_per_row) - (amount_per_row / 2)) * -0.1f + 2f;
            float z = 0f;

            particles[index].position = new Vector3(x, y, z);
        }

        ps.SetParticles(particles, count);
    }

    private void Shift_blank_particles_with_given_position(float[][] coordinates, int count)
    {
        var particles = new ParticleSystem.Particle[count];
        ps.GetParticles(particles);

        for (int index = 0; index < count; index++)
        {
            particles[index].angularVelocity = 0;
            particles[index].angularVelocity3D = new Vector3(0, 0, 0);
            particles[index].velocity = new Vector3(0, 0, 0);
            particles[index].startColor = Color.grey;


            float x = coordinates[index][0];
            float y = coordinates[index][1];
            float z = 0f;

            particles[index].position = (new Vector3(x, y, z));
        }

        ps.SetParticles(particles, count);
    }

    private void Shift_input_particles(double[,] input, int count)
    {
        var particles = new ParticleSystem.Particle[count];
        ps.GetParticles(particles);

        int amount_per_row = Mathf.FloorToInt(Mathf.Sqrt(count));

        var gradient = new Gradient();

        // Blend color from blue at 0% to white at 50% to red at 100%
        var colors = new GradientColorKey[2];
        colors[0] = new GradientColorKey(Color.white, 0.0f);
        colors[1] = new GradientColorKey(Color.black, 1.0f);

        // Keep alphas at 1 for all times
        var alphas = new GradientAlphaKey[2];
        alphas[0] = new GradientAlphaKey(1.0f, 0.0f);
        alphas[1] = new GradientAlphaKey(1.0f, 1f);

        gradient.SetKeys(colors, alphas);

        for (int index = 0; index < count; index++)
        {
            particles[index].angularVelocity = 0;
            particles[index].angularVelocity3D = new Vector3(0, 0, 0);
            particles[index].velocity = new Vector3(0, 0, 0);
            float input_value = (float)input[(int)Mathf.Floor(index / amount_per_row), index % amount_per_row];
            Color color_to_set = gradient.Evaluate(input_value / 255f);
            particles[index].startColor = color_to_set;


            float x = ((index % amount_per_row) - (amount_per_row / 2)) * 0.1f;
            float y = (Mathf.Floor(index / amount_per_row) - (amount_per_row / 2)) * -0.1f + 2f;
            float z = 0f;

            particles[index].position = new Vector3(x, y, z);
        }

        ps.SetParticles(particles, count);
    }

    private void Shift_nap_particles(double[,] naps, int class_index)
    {
        int nap_count = naps.GetLength(1);
        var particles = new ParticleSystem.Particle[nap_count];
        ps.GetParticles(particles);

        double[] class_naps = GetRow(naps, class_index);

        IEnumerable<double> ma = class_naps.Cast<double>();
        //double mean = Statistics.Mean(ma);
        //Debug.Log("Mean is: " + mean.ToString());

        //Matrix<double> matrix = Matrix<double>.Build.DenseOfArray(naps);
      
        // Calculate the mean, min, and max of the matrix
        double meanValue = Statistics.Mean(ma);

        Debug.Log("Old mean value: " + meanValue.ToString());
        //double meanDifference = 0.5 - meanValue;
        //matrix.Add(meanValue);

        double minValue = Statistics.Minimum(ma);
        Debug.Log("Old min value: " + minValue.ToString());
        double maxValue = Statistics.Maximum(ma);
        Debug.Log("Old max value: " + maxValue.ToString());

        // Shift mean to 0.5
        double meanDifference = 0.5 - meanValue;
        double scalingFactor = 1.0 / ((maxValue + meanDifference) - (minValue + meanDifference));

        for (int i = 0; i < nap_count; i++)
        {
            class_naps[i] = (class_naps[i] + meanDifference); //* scalingFactor;
        }

        ma = class_naps.Cast<double>();

        meanValue = Statistics.Mean(ma);

        Debug.Log("New mean value: " + meanValue.ToString());
        //double meanDifference = 0.5 - meanValue;
        //matrix.Add(meanValue);

        minValue = Statistics.Minimum(ma);
        Debug.Log("New min value: " + minValue.ToString());
        maxValue = Statistics.Maximum(ma);
        Debug.Log("New max value: " + maxValue.ToString());

        for (int i = 0; i < nap_count; i++)
        {
            particles[i].angularVelocity = 0;
            particles[i].angularVelocity3D = new Vector3(0, 0, 0);

            particles[i].velocity = new Vector3(0, 0, 0);

            // TODO: Make this dependent on Value of NAP
            //naps[i,j] = naps[i,j] * scalingFactor + offset;
            float nap_value = (float)class_naps[i];
            // Shift nap values between [0,1] to feed it's value to the color gradient

            //nap_value = nap_value * (float)scalingFactor + (float)offset;
            Debug.Log("Scaled NAP value is: " + nap_value.ToString());

            var gradient = new Gradient();

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

            gradient.SetKeys(colors, alphas);

            Color color_to_set = gradient.Evaluate(nap_value);
            particles[i].startColor = color_to_set;

            float x = (i % 10) * 0.1f;
            float y = Mathf.Floor(i / 10) * -0.1f + 2f;
            float z = 0f;

            particles[i].position = new Vector3(x, y, z);
        }
        ps.SetParticles(particles, nap_count);
    }

    private void Shift_WB_particles(double[] biases)
    {
        int count = biases.GetLength(0);
        var particles = new ParticleSystem.Particle[count];
        ps.GetParticles(particles);

        int amount_per_row = Mathf.FloorToInt(Mathf.Sqrt(count));
        double max = biases.Max();
        double min = biases.Min();

        var gradient = new Gradient();

        // Blend color from blue at 0 % to white at 50 % to red at 100 %
        var colors = new GradientColorKey[3];
        colors[0] = new GradientColorKey(Color.blue, 0.0f);
        colors[1] = new GradientColorKey(Color.white, 0.5f);
        colors[2] = new GradientColorKey(Color.red, 1.0f);

        // Keep alphas at 1 for all times
        var alphas = new GradientAlphaKey[3];
        alphas[0] = new GradientAlphaKey(1.0f, 0.0f);
        alphas[1] = new GradientAlphaKey(1.0f, 0.5f);
        alphas[2] = new GradientAlphaKey(1.0f, 1.0f);

        gradient.SetKeys(colors, alphas);

        for (int index = 0; index < count; index++)
        {
            particles[index].angularVelocity = 0;
            particles[index].angularVelocity3D = new Vector3(0, 0, 0);
            particles[index].velocity = new Vector3(0, 0, 0);

            float value = 0f;
            if (biases[index] >= 0)
            {
                value = (float)(biases[index] / max);
            }
            else
            {
                value = (float)(biases[index] / Mathf.Abs((float)min));
    }
            particles[index].startColor = gradient.Evaluate((value + 1) / 2f);

            float x = ((index % amount_per_row) - (amount_per_row / 2)) * 0.1f;
            float y = (Mathf.Floor(index / amount_per_row) - (amount_per_row / 2)) * -0.1f + 2f;
            float z = 0f;

            particles[index].position = new Vector3(x, y, z);
        }
        
        ps.SetParticles(particles, count);
    }

    // TODO: Write Clustering Function that takes the neuron/particle ids and their UMAP Positions and clusters them together.
    public void ClusterParticles(List<int> neuron_ids, float[][] embeddings)
    {
        // Check if there are already requests with given ids going on

        float sum_x = 0f;
        float sum_y = 0f;
        List<int> elementsToRemove = new List<int>();
        // Loop over new ids, figure out their average position, to determine the target position
        foreach (int id in neuron_ids)
        {
            sum_x += embeddings[id][0];
            sum_y += embeddings[id][1];

            for (int index=0; index < clustering_Requests.Count; index++)
            {
                // If same ID, add to removing list
                if (clustering_Requests[index].id == id)
                {
                    elementsToRemove.Add(index);
                }
            }
        }

        // Clean up already exisiting Requests
        elementsToRemove.Sort((a, b) => b.CompareTo(a));
        foreach (int index in elementsToRemove)
        {
            clustering_Requests.RemoveAt(index);
        }

        float target_x = sum_x / neuron_ids.Count;
        float target_y = sum_y / neuron_ids.Count;

        var particles = new ParticleSystem.Particle[ps.particleCount];
        ps.GetParticles(particles);

        ClusteringRequest new_request;
        float start_time = Time.time;

        // Add each id as clustering request with mean position as target position
        // TODO: Check whether the particle.position is local or global space. Would result in Error if global
        foreach (int id in neuron_ids)
        {
            Vector3 current_position = particles[id].position;
            new_request = new ClusteringRequest(id, current_position, new Vector3(target_x, target_y, 0f), start_time);
            clustering_Requests.Add(new_request);
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

// Value Container for a Request of particles to move towards a cluster. 
public class ClusteringRequest
{
    public int id { get; set; }
    public Vector3 start_position { get; set; }
    public Vector3 target_position { get; set; }
    public float start_time { get; set; }

    public ClusteringRequest(int _id, Vector3 _start_position, Vector3 _target_position, float _start_time)
    {
        id = _id;
        start_position = _start_position;
        target_position = _target_position;
        start_time = _start_time;
    }
}
