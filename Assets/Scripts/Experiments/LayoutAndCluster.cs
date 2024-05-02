using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using System.IO;
using Newtonsoft.Json;

public class LayoutAndCluster : MonoBehaviour
{
    public bool start_experiment_trigger = false;
    private bool experiment_running = false;



    private HelloClient hc;
    private DataManager dm;

    // Start is called before the first frame update
    void Start()
    {
        try
        {
            hc = this.gameObject.GetComponent<HelloClient>();
        }
        catch
        {
            Debug.LogError("No HelloClient found on the Gameobject that this experiment is attached to. Please check!");
            return;
        }

        try
        {
            dm = this.gameObject.GetComponent<DataManager>();
        }
        catch
        {
            Debug.LogError("No DataManager found on the Gameobject that this experiment is attached to. Please check!");
            return;
        }
    }

    // Update is called once per frame
    void Update()
    {
        if (start_experiment_trigger)
        {
            start_experiment_trigger = false;
            if (!experiment_running)
            {
                experiment_running = true;
                InitExperiments();
            }
        }
    }

    // Call the ClusterTree To recieve data from the python side
    private void InitExperiments()
    {
        dm.experiment_running = true;
        dm.experiment_function = RunExpermients;
        hc.new_task = HelloRequester.task.send_subset_activations;
        hc.debug_trigger_task = true;

    }

    // Callback function that is provided to the Datamaneger and is called as soon as the datamanager recieves
    // the requested data from the python server
    public void RunExpermients(List<float[][]> data)
    {
        Debug.Log("Recieved the data from the datamanager, could run experiments now");
        
        experiment_running = false;
        dm.experiment_running = false;

        MultipleUMAPRuns(data);
        return;

        Experiment exp = new Experiment();
        exp.Name = "Experiment_test";

        // Clustering on Umap (best case) 
        UmapReduction umap = new UmapReduction();
        Debug.Log("Experiment: Calculating UMAP Layout");
        float[][] embeddings = umap.applyUMAP(data[1], 1.4f);

        Debug.Log("Experiment: Creating cluster Tree on UMAP embeddings");
        ClusterTree ct_umap = new ClusterTree(ConvertFloatArrayToDoubleArray(embeddings));
        (List<double> distances, double total_dist) = TotalDistanceForClustering(ct_umap, embeddings);
        exp.UMAPDistances = distances;
        exp.TotalUMAPDistance = total_dist;
        Debug.Log(total_dist.ToString());

        // Clustering on Real Data
        Debug.Log("Experiment: Creating cluster Tree on real data");
        ClusterTree ct_real = new ClusterTree(ConvertFloatArrayToDoubleArray(data[1]));
        (List<double> distances2, double total_dist2) = TotalDistanceForClustering(ct_real, embeddings);
        exp.RealDistances = distances2;
        exp.TotalRealDistance = total_dist2;
        Debug.Log(total_dist2.ToString());

        // Clustering on Random Data multiple times and average (should be worst case)
        List<List<double>> random_distances = new List<List<double>>();
        List<double> random_average_distance = new List<double>();
        Debug.Log("Experiment: Creating cluster Tree on random data");
        for (int index = 0; index < 10; index++)
        {
            Debug.Log("Starting with random run " + index.ToString());
            float[][] random_data = GenerateRandomArray(data[1].Length, data[1][0].Length);
            ClusterTree ct_random = new ClusterTree(ConvertFloatArrayToDoubleArray(random_data));
            (List<double> distances3, double total_dist3) = TotalDistanceForClustering(ct_random, embeddings);
            random_distances.Add(distances3);
            random_average_distance.Add(total_dist3);
            Debug.Log(total_dist3.ToString());
        }

        exp.RandomDistances = random_distances;
        exp.TotalRandomDistance = random_average_distance;

        string filePath = "Assets/Experiments/experiment.json";

        string jsonData = JsonConvert.SerializeObject(exp, Formatting.Indented);

        // Write the JSON data to the file
        File.WriteAllText(filePath, jsonData);


    }

    private void MultipleUMAPRuns(List<float[][]> data)
    {
        // Need to randomise the random arrays once for all umap runs, so that they are comparable to eachother

        List<Experiment> umap_runs = new List<Experiment>();

        for (int run = 1; run <= 10; run++)
        {
            Experiment exp = new Experiment();
            exp.Name = "Experiment_test";

            // Clustering on Umap (best case) 
            UmapReduction umap = new UmapReduction();
            Debug.Log("Experiment: Calculating UMAP Layout");
            float[][] embeddings = umap.applyUMAP(data[1], 1.4f);

            Debug.Log("Experiment: Creating cluster Tree on UMAP embeddings");
            ClusterTree ct_umap = new ClusterTree(ConvertFloatArrayToDoubleArray(embeddings));
            (List<double> distances, double total_dist) = TotalDistanceForClustering(ct_umap, embeddings);
            exp.UMAPDistances = distances;
            exp.TotalUMAPDistance = total_dist;
            Debug.Log(total_dist.ToString());

            // Clustering on Real Data
            Debug.Log("Experiment: Creating cluster Tree on real data");
            ClusterTree ct_real = new ClusterTree(ConvertFloatArrayToDoubleArray(data[1]));
            (List<double> distances2, double total_dist2) = TotalDistanceForClustering(ct_real, embeddings);
            exp.RealDistances = distances2;
            exp.TotalRealDistance = total_dist2;
            Debug.Log(total_dist2.ToString());

            // Clustering on Random Data multiple times and average (should be worst case)
            List<List<double>> random_distances = new List<List<double>>();
            List<double> random_average_distance = new List<double>();
            Debug.Log("Experiment: Creating cluster Tree on random data");
            for (int index = 0; index < 10; index++)
            {
                Debug.Log("Starting with random run " + index.ToString());
                float[][] random_data = GenerateRandomArray(data[1].Length, data[1][0].Length);
                ClusterTree ct_random = new ClusterTree(ConvertFloatArrayToDoubleArray(random_data));
                (List<double> distances3, double total_dist3) = TotalDistanceForClustering(ct_random, embeddings);
                random_distances.Add(distances3);
                random_average_distance.Add(total_dist3);
                Debug.Log(total_dist3.ToString());
            }

            exp.RandomDistances = random_distances;
            exp.TotalRandomDistance = random_average_distance;

            umap_runs.Add(exp);
        }

        string filePath = "Assets/Experiments/cluster_distances_multiple_umap_runs.json";

        string jsonData = JsonConvert.SerializeObject(umap_runs, Formatting.Indented);

        // Write the JSON data to the file
        File.WriteAllText(filePath, jsonData);
    }

    // Calculates distance of all cluster merging required when going gradually from 1 cluster to max_cluster
    private (List<double>, double) TotalDistanceForClustering(ClusterTree ct, float[][] embeddings)
    {
        int max_cluster = ct.max_size;
        List<double> distances = new List<double>();
        double total_distance = 0.0;

        List <Node> previous_nodes = ct.GetClusters(max_cluster);
        // Iterate through each cluster size
        for (int cluster_amount = max_cluster-1; cluster_amount > 0; cluster_amount--)
        {
            List<Node> nodes = ct.GetClusters(cluster_amount);
            IEnumerable<Node> combined_nodes = previous_nodes.Except(nodes);
            IEnumerable<Node> combined_node = nodes.Except(previous_nodes);

            (float, float) pos_combined_node = GetNodePosition(combined_node.ElementAt(0), embeddings);

            double distance = EucledeanDistance(GetNodePosition(combined_nodes.ElementAt(0), embeddings), pos_combined_node);
            distance += EucledeanDistance(GetNodePosition(combined_nodes.ElementAt(1), embeddings), pos_combined_node);

            distances.Add(distance);
            total_distance += distance;

            previous_nodes = nodes;
        }

        return (distances, total_distance);
    }

    private (float, float) GetNodePosition(Node node, float[][] embeddings)
    {
        float sum_x = 0f;
        float sum_y = 0f;
        // Loop over new ids, figure out their average position, to determine the target position
        foreach (int id in node.neuron_ids)
        {
            sum_x += embeddings[id][0];
            sum_y += embeddings[id][1];
        }
        
        return ((sum_x / node.neuron_ids.Count), (sum_y / node.neuron_ids.Count));
    }

    private double ManhattanDistance((float, float) pos1, (float, float) pos2)
    {
        double result = (double)Mathf.Abs(pos1.Item1 - pos2.Item1);
        result += (double)Mathf.Abs(pos1.Item2 - pos2.Item2);
        return result;

    }

    private double EucledeanDistance((float, float) pos1, (float, float) pos2)
    {
        float sum = Mathf.Pow(pos1.Item1 - pos2.Item1, 2f);
        sum += Mathf.Pow(pos1.Item2 - pos2.Item2, 2f);

        return (double)Mathf.Sqrt(sum);
    }

    private static float[][] GenerateRandomArray(int rows, int columns)
    {
        System.Random rand = new System.Random();
        float[][] random_array = new float[rows][];
        for (int i = 0; i < rows; i++)
        {
            random_array[i] = new float[columns];
            for (int j = 0; j < columns; j++)
            {
                random_array[i][j] = (float)rand.NextDouble();
            }
        }
        return random_array;
    }

    private double[,] ConvertFloatArrayToDoubleArray(float[][] floatArray)
    {
        int numRows = floatArray.Length;
        int numCols = floatArray[0].Length;

        double[,] doubleArray = new double[numRows, numCols];

        for (int i = 0; i < numRows; i++)
        {
            for (int j = 0; j < numCols; j++)
            {
                doubleArray[i, j] = (double)floatArray[i][j];
            }
        }

        return doubleArray;
    }


    // Class to save to the JSON File
    class Experiment
    {
        public string Name { get; set; }
        public List<double> UMAPDistances { get; set; }
        public double TotalUMAPDistance { get; set; }
        public List<double> RealDistances { get; set; }
        public double TotalRealDistance { get; set; }
        public List<List<double>> RandomDistances { get; set; }
        public List<double> TotalRandomDistance { get; set; }

    }
}
