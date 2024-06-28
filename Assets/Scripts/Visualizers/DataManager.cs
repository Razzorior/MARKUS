using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using Unity.VisualScripting;

public class DataManager : MonoBehaviour
{
    public GameObject particle_prefab;
    public GameObject connectivity_prefab;
    public GameObject sphere_prefab;

    // Position from where the first layers particles are supposed to be spawned
    public Vector3 particle_starting_pos = new Vector3(8f, 2f, 3f);
    public Vector3 sphere_starting_pos = new Vector3(15f, 1f, 3f);
    public bool umap_rasterization = false;

    // Stuff for experiments
    public bool experiment_running = false;
    public Action<List<float[][]>> experiment_function;

    // TODO: Write dynamic script. So far this is hard coded for MNIST
    [Range(1, 100)]
    public int hidden_layer_cluster = 100;
    private int last_know_hidden_layer_cluster = 100;
    private bool clustering_and_umap_done = false;
    private List<ClusterTree> hidden_layer_CTs = new List<ClusterTree>();
    private List<float[][]> hidden_layer_embeddings = new List<float[][]>();

    private List<double[,]> naps = null;
    private List<GameObject> particle_objects = new List<GameObject>();
    private List<GameObject> input_particle_objects = new List<GameObject>();
    private List<GameObject> signals_particle_objects = new List<GameObject>();
    private List<GameObject> input_connectivity_objects = new List<GameObject>();
    private List<GameObject> connection_manager_objects = new List<GameObject>();

    // Variables for full class analysis
    [Range(0,9)]
    public int class_index = 0;
    private int last_know_class_index = 0;
    private List<List<float[]>> class_average_activations;
    private List<List<float[][]>> class_average_signals;
    private bool class_analysis_running = false;
    public bool class_normalized_lines = true;
    private bool last_known_class_normalized_lines = true;
    private List<(double, double)> scales_per_layer = new List<(double, double)>();

    // Variables for Signal Lines
    public bool show_max_lines = true;
    private bool last_known_show_max_lines = true;
    [Range(1,1000)]
    public int max_lines = 100;
    private int last_known_max_lines = 100;
    public bool backwards_pass = false;
    private bool last_known_backwards_pass = false;

    // Start is called before the first frame update
    // TODO: add the last_known variables into here, so that they are loaded on start, in cases the original variables have been changed in inspector
    void Start()
    {
        if (particle_prefab is null)
        {
            Debug.Log("Warning: particle_prefab not set in inspector!");
        }
    }

    // Update is called once per frame
    void Update()
    {
        UpdateClassSlider();
        UpdateClusterSlider();
        UpdateMaxLinesSlider();
        UpdateToggles();
        bool update_done = UpdateParticles();
        if (update_done == true) { UpdateLines(); }
    }

    private void UpdateClusterSlider()
    {
        // BEGIN: Clustering slider
        if (clustering_and_umap_done == false)
        {
            return;
        }

        if (last_know_hidden_layer_cluster == hidden_layer_cluster)
        {
            return;
        }

        // if here, there is a new amount of clusters requested.
        for (int index = 0; index < hidden_layer_CTs.Count; index++)
        {
            int amount_of_clusters = Mathf.RoundToInt(hidden_layer_CTs[index].max_size * (hidden_layer_cluster / 100f));
            List<Node> clusters = hidden_layer_CTs[index].GetClusters(amount_of_clusters);

            // TODO: This is hardcoded for simple MLP. Needs to be generalized eventually
            ParticleManager pm = signals_particle_objects[index+1].GetComponent<ParticleManager>();

            foreach (Node cluster in clusters)
            {
                pm.ClusterParticles(cluster.neuron_ids, hidden_layer_embeddings[index]);
            }
        }

        // After all is done, set remembered layer
        last_know_hidden_layer_cluster = hidden_layer_cluster;
        // END: Clustering slider
    }

    private void UpdateClassSlider()
    {
        if (class_analysis_running == false)
        {
            return;
        }

        if(last_know_class_index == class_index) 
        {
            return;
        }

        if (!backwards_pass)
            Rebuild_ClassAnalysis_Lines();
        else
            BuildBackwardPassLines();

        last_know_class_index = class_index;
    }

    private void UpdateMaxLinesSlider()
    {
        if (class_analysis_running == false)
        {
            return;
        }

        if (last_known_max_lines == max_lines)
        {
            return;
        }

        if (backwards_pass)
        {
            BuildBackwardPassLines();
        }
        else
        {
            Rebuild_ClassAnalysis_Lines();
        }

        last_known_max_lines = max_lines;
    }

    // Function with all the different boolean toggles for the visualization. Add all toggle functions here.
    private void UpdateToggles()
    {
        UpdateClassNormalizedLines();
        UpdateMaxMinLines();
        ToggleBackwardsPass();
    }

    private void UpdateClassNormalizedLines()
    {
        // No change on button, return
        if (class_normalized_lines == last_known_class_normalized_lines) return;
        // No class analysis -> Not able to change either. Ignore the request, so return
        if (class_analysis_running == false) return;

        // Makes sure that scales_per_layer is only computed once and then stored. 
        if (!class_normalized_lines && scales_per_layer.Count == 0)
        {
            scales_per_layer = FindMinMax(class_average_signals);
        }

        if (backwards_pass)
        {
            BuildBackwardPassLines();
        }
        else
        { 
            Rebuild_ClassAnalysis_Lines(); 
        }

        last_known_class_normalized_lines = class_normalized_lines;
    }

    private void UpdateMaxMinLines()
    {
        // No change of the bool, so nothing to do -> return
        if (show_max_lines == last_known_show_max_lines) return;

        // No class analysis -> Not implemented yet. TODO
        if (class_analysis_running == false) return;
        // Min Lines doesn't make sence for backward pass, so ignore it. 
        if (backwards_pass) return;

        Rebuild_ClassAnalysis_Lines();

        last_known_show_max_lines = show_max_lines;
    }

    private void ToggleBackwardsPass()
    {
        if (class_analysis_running == false) return;

        if (backwards_pass == last_known_backwards_pass) return;

        // Turn Backward pass view off.
        if (!backwards_pass)
        {
            Rebuild_ClassAnalysis_Lines();
            ClearHighlightedParticles();
        }
        // Turn Backward pass view on.
        else
        {
            BuildBackwardPassLines();
        }

        last_known_backwards_pass = backwards_pass;
    }

    private bool UpdateParticles()
    {
        if (signals_particle_objects.Count <= 0)
        {
            return false;
        }

        bool res = false;
        for (int index = 0; index < hidden_layer_CTs.Count; index++)
        {
            res = signals_particle_objects[index+1].GetComponent<ParticleManager>().DataManagerUpdate();
        }
        return res;
    }

    private void UpdateLines()
    {
        if (connection_manager_objects.Count <=0)
        {
            return;
        }

        foreach (GameObject go in connection_manager_objects)
        {
            go.GetComponent<ConnectionManager>().DataManagerUpdate();
        }

    }

    public void CleanUpForNewLoadedModel()
    {
        DeleteExistingLines();
        DeleteExistingParticles();
        ResetOtherParameters();
    }

    public void InitWeightedActivationLines(List<double[,]> weighted_activations)
    {
        if (input_particle_objects.Count <= 0)
        {
            Debug.LogError("No particle-Systems to connect the weighted activations found!");
            return;
        }

        if (input_connectivity_objects.Count > 0)
        {
            // Make sure that dimensions are matching.
            if (weighted_activations.Count != input_connectivity_objects.Count)
            {
                Debug.LogError("The amount of weighted activations does not match the amount of drawn connectivity layers!");
                return;
            }

            foreach (GameObject obj in input_connectivity_objects)
            {
                Destroy(obj);
            }

            input_connectivity_objects.Clear();

        }

        for (int index = 0; index < weighted_activations.Count; index++)
        {
            GameObject connectivity_go = Instantiate(connectivity_prefab, Vector3.zero, Quaternion.identity);

            // Adding the particle systems auf the two layers to connect.
            List<ParticleSystem> particle_systems = new List<ParticleSystem>();
            particle_systems.Add(input_particle_objects[index].GetComponent<ParticleSystem>());
            particle_systems.Add(input_particle_objects[index + 1].GetComponent<ParticleSystem>());

            connectivity_go.GetComponent<ConnectionManager>().InitFixedConnectivity(weighted_activations[index], 100, particle_systems, input_particle_objects[index].transform, input_particle_objects[index + 1].transform);
            input_connectivity_objects.Add(connectivity_go);
        }


    }

    public void InitParticleManagerForInput(double[,] input, List<double[]> activations)
    {
        Debug.Log("InitParticleManagerForInput was called");
        if (input_particle_objects.Count > 0)
        {
            UpdateParticleManagerForInput(input, activations);
            HelloClient cl = this.gameObject.GetComponent<HelloClient>();
            cl.new_task = HelloRequester.task.send_weighted_activations;
            cl.debug_trigger_task = true;
            return;
        }

        GameObject go = Instantiate(particle_prefab, particle_starting_pos + new Vector3(6f, 0f, 0f), Quaternion.identity);
        int total_input_count = input.GetLength(0) * input.GetLength(1);
        go.GetComponent<ParticleManager>().InitParticleSystemsInput(input, total_input_count);

        input_particle_objects.Add(go);

        for (int index = 0; index < activations.Count; index++)
        {
            go = Instantiate(particle_prefab, particle_starting_pos - new Vector3(-6f, 0f, (index + 1f) * 2f), Quaternion.identity);
            go.GetComponent<ParticleManager>().InitParticleSystemsWB(activations[index]);
            input_particle_objects.Add(go);
        }

        HelloClient used_client = this.gameObject.GetComponent<HelloClient>();
        used_client.new_task = HelloRequester.task.send_weighted_activations;
        used_client.debug_trigger_task = true;
    }

    public void UpdateParticleManagerForInput(double[,] input, List<double[]> activations)
    {
        input_particle_objects[0].GetComponent<ParticleManager>().UpdateParticleSystemsInput(input);

        for (int index = 1; index < input_particle_objects.Count; index++)
        {
            input_particle_objects[index].GetComponent<ParticleManager>().UpdateParticleSytemsWB(activations[index - 1]);
        }
    }

    /// <summary>
    /// Creates Prefab containing ParticleManager.cs and Particle System component for each layer and passes
    /// the corrosponding naps. Is per default at class index 0.
    /// </summary>
    /// <param name="naps">The NAPs arrays per layer stored in a List</param>
    /// <param name="class_index">The class index of the classifier</param>
    public void InitParticleManagerForNaps(List<double[,]> naps, int class_index = 0)
    {
        this.naps = naps;

        foreach (double[,] nap_layer in naps)
        {
            int classes_count = nap_layer.GetLength(0);
            GameObject go = Instantiate(particle_prefab, particle_starting_pos, Quaternion.identity);
            go.GetComponent<ParticleManager>().InitParticleSystemsNAPs(nap_layer, class_index);
        }
    }

    public void InitParticleManagerMLP(List<double[,]> weights, List<double[]> biases)
    {
        GameObject go = Instantiate(particle_prefab, particle_starting_pos, Quaternion.identity);
        int first_layer_neuron_count = weights[0].GetLength(0);

        go.GetComponent<ParticleManager>().InitParticleSystems(first_layer_neuron_count);
        particle_objects.Add(go);
        /*Debug.Log("Weights dims:");
        foreach (double[,] weight_layer in weights)
        {
            Debug.Log(weight_layer.GetLength(0));
            Debug.Log(weight_layer.GetLength(1));
        }
        Debug.Log("Bias dims:");*/
        for (int bias_index = 0; bias_index < biases.Count; bias_index++)
        {
            go = Instantiate(particle_prefab, particle_starting_pos - new Vector3(0f, 0f, (bias_index + 1f) * 2f), Quaternion.identity);
            go.GetComponent<ParticleManager>().InitParticleSystemsWB(biases[bias_index]);
            particle_objects.Add(go);
        }


        for (int weight_index = 0; weight_index < weights.Count; weight_index++)
        {
            GameObject connectivity_go = Instantiate(connectivity_prefab, Vector3.zero, Quaternion.identity);

            // Adding the particle systems auf the two layers to connect.
            List<ParticleSystem> particle_systems = new List<ParticleSystem>();
            particle_systems.Add(particle_objects[weight_index].GetComponent<ParticleSystem>());
            particle_systems.Add(particle_objects[weight_index+1].GetComponent<ParticleSystem>());

            connectivity_go.GetComponent<ConnectionManager>().InitFixedConnectivity(weights[weight_index], 100, particle_systems, particle_objects[weight_index].transform, particle_objects[weight_index+1].transform);
        }

    }

    public void InitSphereManager(double[,] input, double[,] ig)
    {
        GameObject go = Instantiate(sphere_prefab, sphere_starting_pos, Quaternion.identity);
        go.GetComponent<SphereManager>().InitIGVisualization(input, ig);
    }

    public void InitClusteringManager(List<double[]> activations)
    {
        ClusterTree ct = new ClusterTree(activations[1]);
        List<Node> clusters = ct.GetClusters(10);

        int cluster_index = 1;
        foreach (Node node in clusters)
        {
            Debug.Log("Cluster " + cluster_index.ToString() + " with the following neuron IDs: ");
            string str = "";
            foreach (int i in node.neuron_ids)
            {
                str += (i.ToString() + " ");
            }
            Debug.Log(str);
            cluster_index++;
        }
    }

    public void InitUmapLayout(List<float[][]> data_array, bool is_signals = false)
    {
        if (experiment_running == true)
        {
            experiment_function(data_array);
            return;
        }

        UmapReduction umap = new UmapReduction();
        // Amount of layers, excluding the output layer.
        int amount_of_layers = data_array.Count;
        Debug.Log("Amount of layers: " + amount_of_layers);

        // Init particles of InputLayer
        int input_layer_size = data_array[0].Length;
        float[][] input_layer_coordinates = new float[input_layer_size][];

        int amount_per_row = Mathf.FloorToInt(Mathf.Sqrt(input_layer_size));
        for (int index = 0; index < input_layer_size; index++)
        {
            float x = ((index % amount_per_row) - (amount_per_row / 2)) * 0.1f;
            float y = (Mathf.Floor(index / amount_per_row) - (amount_per_row / 2)) * -0.1f;

            input_layer_coordinates[index] = new float[] { x, y };
        }

        GameObject go = Instantiate(particle_prefab, particle_starting_pos + new Vector3(12f, 0f, 0f), Quaternion.Euler(0, 90, 0) * Quaternion.identity);
        go.GetComponent<ParticleManager>().InitParticleSystemsWithGivenPositions(input_layer_coordinates);
        signals_particle_objects.Add(go);

        // TODO: Rasterize hidden layers using UMAP reduction and init their particles
        int lim;
        int hidden_index;
        if (is_signals)
        {
            lim = amount_of_layers;
            hidden_index = 1;
        }
        else
        {
            lim = amount_of_layers - 1;
            hidden_index = 1;
        }
        for (int index = hidden_index; index < lim; index++)
        {
            // TODO: Combine Input and output signals of hidden layer and use umap reduction to 2D for Signals
            float[][] embeddings = umap.applyUMAP(data_array[index], 1.4f);

            if (umap_rasterization == true)
            {
                //TODO: Implement rasterization
            }
            else
            {
                Debug.Log("Creating hidden layer");
                GameObject _tmp = Instantiate(particle_prefab, particle_starting_pos + new Vector3(12f, 0f, 0f)
                    + new Vector3(3*(index + (1-hidden_index)), 0f, 0f), Quaternion.Euler(0, 90, 0) * Quaternion.identity);
                _tmp.GetComponent<ParticleManager>().InitParticleSystemsWithGivenPositions(embeddings);
                signals_particle_objects.Add(_tmp);
            }
            hidden_layer_embeddings.Add(embeddings);
            hidden_layer_CTs.Add(new ClusterTree(ConvertFloatArrayToDoubleArray(data_array[index])));
        }

        // Init Output Layer Particles
        int output_layer_size;
        if (is_signals)
        {
            output_layer_size = data_array[data_array.Count - 1][0].Length;
        }
        else
        {
            output_layer_size = data_array[data_array.Count - 1].Length;
        }

        float[][] output_layer_coordinates = new float[output_layer_size][];

        for(int index = 0; index < output_layer_size; index++)
        {
            float x = ((index % output_layer_size) - (output_layer_size / 2)) * 0.1f;
            float y = 0f;

            output_layer_coordinates[index] = new float[] { x, y };
        }

        go = Instantiate(particle_prefab, particle_starting_pos + new Vector3(12f, 0f, 0f)
                    + new Vector3(3 * data_array.Count, 0f, 0f), Quaternion.Euler(0, 90, 0) * Quaternion.identity);
        go.GetComponent<ParticleManager>().InitParticleSystemsWithGivenPositions(output_layer_coordinates);
        signals_particle_objects.Add(go);

        if (is_signals)
        {
            // Draw Signal Lines (weighted activations)
            for (int index = 0; index < data_array.Count; index++)
            {
                GameObject connectivity_go = Instantiate(connectivity_prefab, Vector3.zero, Quaternion.identity);

                // Adding the particle systems auf the two layers to connect.
                List<ParticleSystem> particle_systems = new List<ParticleSystem>();
                particle_systems.Add(signals_particle_objects[index].GetComponent<ParticleSystem>());
                particle_systems.Add(signals_particle_objects[index + 1].GetComponent<ParticleSystem>());

                connectivity_go.GetComponent<ConnectionManager>().InitFixedConnectivity(ConvertFloatArrayToDoubleArray(data_array[index])
                    , 100, particle_systems, signals_particle_objects[index].transform, signals_particle_objects[index + 1].transform);

                connection_manager_objects.Add(connectivity_go);
            }
        }

        //TODO: Init Clustering for any hidden layer 
        ClusterTree ct = new ClusterTree(ConvertFloatArrayToDoubleArray(data_array[1]));
        //ClusterTree ct = new ClusterTree(ConvertFloatArrayToDoubleArray(umap.CombineArrays(average_signals[0], average_signals[1])));


        /*//TODO: Figure out what happend with the multidim cluster tree
        List<Node> clusters = ct.GetClusters(10);

        int cluster_index = 1;
        foreach (Node node in clusters)
        {
            Debug.Log("Cluster " + cluster_index.ToString() + " with the following neuron IDs: ");
            string str = "";
            foreach (int i in node.neuron_ids)
            {
                str += (i.ToString() + " ");
            }
            Debug.Log(str);
            cluster_index++;
        }*/
        clustering_and_umap_done = true;

        Debug.Log("Done with cluster computation");
        
    }

    public void InitUmapLayoutWithFullData(List<float[]> _activations, List<float[][]> _subset_activations, List<float[][]> _signals)
    {
        UmapReduction umap = new UmapReduction();
        int amount_of_layers = _activations.Count; // Helper: 4

        // Init particles of InputLayer
        int input_layer_size = _activations[0].Length;
        float[][] input_layer_coordinates = new float[input_layer_size][];

        int amount_per_row = Mathf.FloorToInt(Mathf.Sqrt(input_layer_size));
        for (int index = 0; index < input_layer_size; index++)
        {
            float x = ((index % amount_per_row) - (amount_per_row / 2)) * 0.1f;
            float y = (Mathf.Floor(index / amount_per_row) - (amount_per_row / 2)) * -0.1f;

            input_layer_coordinates[index] = new float[] { x, y };
        }

        GameObject go = Instantiate(particle_prefab, particle_starting_pos + new Vector3(12f, 0f, 0f), Quaternion.Euler(0, 90, 0) * Quaternion.identity);
        go.GetComponent<ParticleManager>().InitParticleSystemsWithGivenPositions(input_layer_coordinates);
        signals_particle_objects.Add(go);

        for (int index = 1; index < _activations.Count - 1; index++)
        {
            float[][] embeddings = umap.applyUMAP(_subset_activations[index], 1.4f);

            GameObject _tmp = Instantiate(particle_prefab, particle_starting_pos + new Vector3(12f, 0f, 0f)
                + new Vector3(3 * index, 0f, 0f), Quaternion.Euler(0, 90, 0) * Quaternion.identity);
            _tmp.GetComponent<ParticleManager>().InitParticleSystemsWithGivenPositions(embeddings);
            signals_particle_objects.Add(_tmp);
            
            hidden_layer_embeddings.Add(embeddings);
            hidden_layer_CTs.Add(new ClusterTree(ConvertFloatArrayToDoubleArray(_subset_activations[index])));
        }

        // Init Output Layer Particles
        int output_layer_size = _activations[_activations.Count - 1].Length;

        float[][] output_layer_coordinates = new float[output_layer_size][];

        for (int index = 0; index < output_layer_size; index++)
        {
            float x = ((index % output_layer_size) - (output_layer_size / 2)) * 0.1f;
            float y = 0f;

            output_layer_coordinates[index] = new float[] { x, y };
        }

        // TODO: Don't use InitParticleSystemWB, build a new function with similar functionality but custom positioning of the particles,
        // TODO: or change the way InitParticleSystemsWB takes those in.
        go = Instantiate(particle_prefab, particle_starting_pos + new Vector3(12f, 0f, 0f)
                    + new Vector3(3 * (_activations.Count - 1), 0f, 0f), Quaternion.Euler(0, 90, 0) * Quaternion.identity);
        go.GetComponent<ParticleManager>().InitParticleSystemsWithGivenPositions(output_layer_coordinates);
        signals_particle_objects.Add(go);


        // Init Signal Lines
        for (int index = 0; index < _signals.Count; index++)
        {
            GameObject connectivity_go = Instantiate(connectivity_prefab, Vector3.zero, Quaternion.identity);

            // Adding the particle systems auf the two layers to connect.
            List<ParticleSystem> particle_systems = new List<ParticleSystem>();
            particle_systems.Add(signals_particle_objects[index].GetComponent<ParticleSystem>());
            particle_systems.Add(signals_particle_objects[index + 1].GetComponent<ParticleSystem>());

            connectivity_go.GetComponent<ConnectionManager>().InitFixedConnectivity(ConvertFloatArrayToDoubleArray(_signals[index])
                , 100, particle_systems, signals_particle_objects[index].transform, signals_particle_objects[index + 1].transform);

            connection_manager_objects.Add(connectivity_go);
        }
        
        clustering_and_umap_done = true;
    }

    public void InitFullAnalysisOfClass(List<List<float[]>> _class_average_activations, List<float[][]> _subset_activations, List<List<float[][]>> _class_average_signals )
    {
        class_average_activations = _class_average_activations;
        class_average_signals = _class_average_signals;
        class_analysis_running = true;
        InitUmapLayoutWithFullData(class_average_activations[class_index], _subset_activations, class_average_signals[class_index]);
    }

    public void Show_Different_ClassNAPs(int class_index)
    {
        foreach (double[,] nap_layer in this.naps)
        {
            int classes_count = nap_layer.GetLength(0);
        }
    }

    private void BuildBackwardPassLines()
    {
        List<int> ids = new List<int> { class_index };
        DeleteExistingLines();


        signals_particle_objects[class_average_signals[class_index].Count].GetComponent<ParticleManager>().HighlightGivenParticlesAndGreyRest(new List<int> { class_index });

        for (int index = class_average_signals[class_index].Count - 1; index >= 0; index--)
        {
            GameObject connectivity_go = Instantiate(connectivity_prefab, Vector3.zero, Quaternion.identity);

            // Adding the particle systems auf the two layers to connect.
            List<ParticleSystem> particle_systems = new List<ParticleSystem>();
            particle_systems.Add(signals_particle_objects[index].GetComponent<ParticleSystem>());
            particle_systems.Add(signals_particle_objects[index + 1].GetComponent<ParticleSystem>());

            int line_amount = (int)Math.Round(max_lines / 10.0, MidpointRounding.AwayFromZero);
            if (line_amount <= 0)
            {
                line_amount = 1;
            }

            if (class_normalized_lines)
            {
                ids = connectivity_go.GetComponent<ConnectionManager>().InitBackwardsPassOfLayers(ConvertFloatArrayToDoubleArray(class_average_signals[class_index][index])
                , line_amount, particle_systems, signals_particle_objects[index].transform, signals_particle_objects[index + 1].transform, ids);
            }
            else
            {
                ids = connectivity_go.GetComponent<ConnectionManager>().InitBackwardsPassOfLayers(ConvertFloatArrayToDoubleArray(class_average_signals[class_index][index])
                , line_amount, particle_systems, signals_particle_objects[index].transform, signals_particle_objects[index + 1].transform, ids, scales_per_layer[index]);
            }
            connection_manager_objects.Add(connectivity_go);

            signals_particle_objects[index].GetComponent<ParticleManager>().HighlightGivenParticlesAndGreyRest(ids);
        }
    }

    private void Rebuild_ClassAnalysis_Lines()
    {
        DeleteExistingLines();

        for (int index = 0; index < class_average_signals[class_index].Count; index++)
        {
            GameObject connectivity_go = Instantiate(connectivity_prefab, Vector3.zero, Quaternion.identity);

            // Adding the particle systems auf the two layers to connect.
            List<ParticleSystem> particle_systems = new List<ParticleSystem>();
            particle_systems.Add(signals_particle_objects[index].GetComponent<ParticleSystem>());
            particle_systems.Add(signals_particle_objects[index + 1].GetComponent<ParticleSystem>());

            if (class_normalized_lines)
            {
                connectivity_go.GetComponent<ConnectionManager>().InitFixedConnectivity(ConvertFloatArrayToDoubleArray(class_average_signals[class_index][index])
                    , max_lines, particle_systems, signals_particle_objects[index].transform, signals_particle_objects[index + 1].transform, use_max_values: show_max_lines);

            }
            else
            {
                connectivity_go.GetComponent<ConnectionManager>().InitFixedConnectivity(ConvertFloatArrayToDoubleArray(class_average_signals[class_index][index])
                    , max_lines, particle_systems, signals_particle_objects[index].transform, signals_particle_objects[index + 1].transform, scales_per_layer[index], use_max_values: show_max_lines);
            }

            connection_manager_objects.Add(connectivity_go);
        }
    }

    private void ClearHighlightedParticles()
    {
        List<int> empty_list = new List<int>();

        foreach (GameObject go in signals_particle_objects)
        {
            go.GetComponent<ParticleManager>().HighlightGivenParticlesAndGreyRest(empty_list);
        }
    }

    private void DeleteExistingLines()
    {
        for (int index = 0; index < connection_manager_objects.Count; index++)
        {
            GameObject.Destroy(connection_manager_objects[index]);
        }
        connection_manager_objects = new List<GameObject>();

        for (int index = 0; index < input_connectivity_objects.Count; index++)
        {
            GameObject.Destroy(input_connectivity_objects[index]);
        }
        input_connectivity_objects = new List<GameObject>();
    }

    private void DeleteExistingParticles()
    {
        for (int index = 0; index < particle_objects.Count; index++)
        {
            GameObject.Destroy(particle_objects[index]);
        }
        particle_objects = new List<GameObject>();

        for (int index = 0; index < input_particle_objects.Count; index++)
        {
            GameObject.Destroy(input_particle_objects[index]);
        }
        input_particle_objects = new List<GameObject>();

        for (int index = 0; index < signals_particle_objects.Count; index++)
        {
            GameObject.Destroy(signals_particle_objects[index]);
        }
        signals_particle_objects = new List<GameObject>();
    }

    private void ResetOtherParameters()
    {
        hidden_layer_CTs = new List<ClusterTree>();
        hidden_layer_embeddings = new List<float[][]>();
        naps = null;
        clustering_and_umap_done = false;
        class_analysis_running = false;
        class_average_activations = new List<List<float[]>>();
        class_average_signals = new List<List<float[][]>>();
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

    private double[] ConvertFloatArrToDoubleArr(float[] floatArray)
    {
        int arr_size = floatArray.Length;

        double[] doubleArray = new double[arr_size];
        for (int i = 0; i < arr_size; i++)
        {
            doubleArray[i] = (double)floatArray[i];
        }

        return doubleArray;
    }

    public List<(double min, double max)> FindMinMax(List<List<float[][]>> my_list)
    {
        // The amount of layers in the model
        int layer_amount = my_list[0].Count;
        // The amount of classes in the data
        int class_amount = my_list.Count;

        // List that will be returned
        List<(double, double)> result = new List<(double, double)>();

        // Iterate through the layer first, because we want min max for each layer through all classes
        for (int layer_index = 0; layer_index < layer_amount; layer_index++)
        {
            // init min max with the opposite max min value;
            float min = float.MaxValue;
            float max = float.MinValue;

            for (int class_index = 0; class_index < class_amount; class_index++)
            {
                foreach (var array in my_list[class_index][layer_index])
                {
                    foreach (var element in array)
                    {
                        min = Mathf.Min(min, element);
                        max = Mathf.Max(max, element);
                    }

                }  
            }
            result.Add((min, max));
        }

        return result;
    }
}