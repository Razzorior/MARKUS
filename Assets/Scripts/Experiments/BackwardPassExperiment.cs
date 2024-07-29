using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using System.IO;
using Newtonsoft.Json;
using UnityEngine.Assertions.Must;
using static DataManager;
using System;

public class BackwardspassExperiment : MonoBehaviour
{
    public bool start_experiment_trigger = false;
    private bool experiment_running = false;

    private HelloClient hc;
    private DataManager dm;

    private List<string> model_names_to_load = new List<string>() { "two_layer_mlp_net", "two_layer_mlp_underfitted", "two_layer_mlp_untrained" };
    private int iteration_amounts = 10;
    private int current_model_index = 0;
    private int current_iteration = 0;

    private DataManager.UMAPData current_data_source = 0;

    private List<Experiment> experiments = new List<Experiment>();
    
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
        // Trigger was pressed -> Start experiment
        if (start_experiment_trigger)
        {
            start_experiment_trigger = false;
            if (!experiment_running)
            {
                InitExperiments();
            }
        }
        // Experiment was already called once, do it again until all runs are done.
        if (experiment_running)
        {
            InitExperiments();
        }
    }

    // Inits Request to the Python Server. Changing dm.experiment_running to true ensures that the RunExperiments function will be called, as soon as the data has been
    // transmitted and prepared.
    private void InitExperiments()
    {
        dm.experiment_running = true;
        dm.experiment_function2 = RunExpermients;
        hc.new_task = HelloRequester.task.load_model_and_send_class_analysis_data;
        int model_index_to_load = FindIndexOfModelName();

        // Check if model index was found or not. If not, display error and return 
        if (model_index_to_load == -1)
        {
            Debug.LogError($"Task {model_names_to_load[current_model_index]} not found!");
            return;
        }

        dm.data_source_for_umap = current_data_source;

        hc.model_index = model_index_to_load;
        hc.debug_trigger_task = true;

        // Turn this off, so that the request won't be called in every update Frame
        if (experiment_running)
            experiment_running = false;
    }

    // Callback function that is provided to the Datamaneger and is called as soon as the datamanager recieves
    // the requested data from the python server
    public void RunExpermients(List<List<List<Vector3>>> positions_of_highlighted_neurons)
    {
        experiment_running = true;
        Experiment exp = new Experiment();
        // Name structure example: two_layer_mlp_net_iteration_0
        exp.Name = Enum.GetName(typeof(UMAPData), current_data_source) + "_iteration_" + current_iteration.ToString();
        exp.PositionsPerHiddenLayer = ConvertToCoordinates(positions_of_highlighted_neurons[0]);

        List<float> distances = new List<float>();
        // Iterate through all layers to calculate total distances between highlighted particles to eachother
        foreach (List<Vector3> layer_positions in positions_of_highlighted_neurons[0])
        {
            distances.Add(CalculateTotalDistancesToEachother(layer_positions));
        }

        exp.TotalDistancesToEachother = distances;

        Debug.Log($"{Enum.GetName(typeof(UMAPData), current_data_source)} run with distance: {distances[0]}");

        distances = new List<float>();
        foreach (List < Vector3 > layer_positions in positions_of_highlighted_neurons[1])
        {
            distances.Add(CalculateTotalDistancesToEachother(layer_positions));
        }

        exp.TotalDistancesAllToEachother = distances;

        Debug.Log($"{Enum.GetName(typeof(UMAPData), current_data_source)} run with all neurons distance: {distances[0]}");


        // Done with iteration of model, save the exp to the list, to save it later
        experiments.Add(exp);

        Debug.Log($"Done with {Enum.GetName(typeof(UMAPData), current_data_source)}'s iteration nbr. " + current_iteration.ToString());

        int maxEnumValue = Enum.GetValues(typeof(UMAPData)).Cast<int>().Max();

        // Check if the last iteration was done, if so change the model that needs to be loaded. 
        if (current_iteration + 1 >= iteration_amounts)
        {
            // If the last iteration of the last model was done, save the experiment
            if ((int)current_data_source + 1 > maxEnumValue)
            {
                SaveExperiment();
            }
            else
            {
                current_data_source++;
                current_iteration = 0;
            }
        }
        // If not last iteration, then continue with the next iteration, which will be called on the next Update()
        else
        {
            current_iteration++;
        }
    }

    private void SaveExperiment()
    {
        Debug.Log("Done with calculations. Saving the Experiment to .json.");
        

        string filePath = "Assets/Experiments/BackwardsPassExperimentDataSource.json";
        string jsonData = JsonConvert.SerializeObject(experiments, Formatting.Indented);

        // Write the JSON data to the file
        File.WriteAllText(filePath, jsonData);

        dm.experiment_running = false;
        experiment_running = false;
    }

    private int FindIndexOfModelName()
    {
        return hc.models_available.IndexOf(model_names_to_load[current_model_index]);
    }

    private float CalculateTotalDistancesToEachother(List<Vector3> positions)
    {
        float totalDistance = 0;

        // Iterate through all positions in the List
        for (int position_index_1 = 0; position_index_1 < positions.Count-1; position_index_1++)
        {
            // Start with position_index_1 + 1 to not compare twice and not with itself.
            for (int position_index_2 = position_index_1+1; position_index_2 < positions.Count; position_index_2++)
            {
                // Add Distance between these two points to the total distance
                totalDistance += Vector3.Distance(positions[position_index_1], positions[position_index_2]);
            }
        }

        return totalDistance / ((positions.Count * (positions.Count - 1)) / 2);
    }

    private List<List<Coordinate>> ConvertToCoordinates(List<List<Vector3>> vectorLists)
    {
        var coordinateLists = new List<List<Coordinate>>();

        foreach (var vectorList in vectorLists)
        {
            var coordinateList = new List<Coordinate>();

            foreach (var vector in vectorList)
            {
                var coordinate = new Coordinate
                {
                    x = vector.x,
                    y = vector.y,
                    z = vector.z
                };

                coordinateList.Add(coordinate);
            }

            coordinateLists.Add(coordinateList);
        }

        return coordinateLists;
    }

    // Class to save to the JSON File
    class Experiment
    {
        public string Name { get; set; }
        public List<List<Coordinate>> PositionsPerHiddenLayer { get; set; }
        public List<float> TotalDistancesToEachother { get; set; }
        public List<float> TotalDistancesAllToEachother { get; set; }

    }

    class Coordinate
    {
        public float x { get; set; }
        public float y { get; set; }
        public float z { get; set; }
    }

    /* 
     * 
     * OLD EXPERIMENT. STORED IN CASE WE NEED THIS AGAIN
     * 
     *  private List<string> model_names_to_load = new List<string>() { "two_layer_mlp_net", "two_layer_mlp_underfitted", "two_layer_mlp_untrained" };
    private int iteration_amounts = 10;
    private int current_model_index = 0;
    private int current_iteration = 0;

    private List<Experiment> experiments = new List<Experiment>();
    
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
        // Trigger was pressed -> Start experiment
        if (start_experiment_trigger)
        {
            start_experiment_trigger = false;
            if (!experiment_running)
            {
                InitExperiments();
            }
        }
        // Experiment was already called once, do it again until all runs are done.
        if (experiment_running)
        {
            InitExperiments();
        }
    }

    // Inits Request to the Python Server. Changing dm.experiment_running to true ensures that the RunExperiments function will be called, as soon as the data has been
    // transmitted and prepared.
    private void InitExperiments()
    {
        dm.experiment_running = true;
        dm.experiment_function2 = RunExpermients;
        hc.new_task = HelloRequester.task.load_model_and_send_class_analysis_data;
        int model_index_to_load = FindIndexOfModelName();

        // Check if model index was found or not. If not, display error and return 
        if (model_index_to_load == -1)
        {
            Debug.LogError($"Task {model_names_to_load[current_model_index]} not found!");
            return;
        }

        hc.model_index = model_index_to_load;
        hc.debug_trigger_task = true;

        // Turn this off, so that the request won't be called in every update Frame
        if (experiment_running)
            experiment_running = false;
    }

    // Callback function that is provided to the Datamaneger and is called as soon as the datamanager recieves
    // the requested data from the python server
    public void RunExpermients(List<List<Vector3>> positions_of_highlighted_neurons)
    {
        experiment_running = true;
        Experiment exp = new Experiment();
        // Name structure example: two_layer_mlp_net_iteration_0
        exp.Name = model_names_to_load[current_model_index] + "_iteration_" + current_iteration.ToString();
        exp.PositionsPerHiddenLayer = ConvertToCoordinates(positions_of_highlighted_neurons);

        List<float> distances = new List<float>();
        // Iterate through all layers to calculate total distances between highlighted particles to eachother
        foreach (List<Vector3> layer_positions in positions_of_highlighted_neurons)
        {
            distances.Add(CalculateTotalDistancesToEachother(layer_positions));
        }

        exp.TotalDistancesToEachother = distances;

        // Done with iteration of model, save the exp to the list, to save it later
        experiments.Add(exp);

        Debug.Log($"Done with {model_names_to_load[current_model_index]}'s iteration nbr. " + current_iteration.ToString());

        // Check if the last iteration was done, if so change the model that needs to be loaded. 
        if (current_iteration + 1 >= iteration_amounts)
        {
            // If the last iteration of the last model was done, save the experiment
            if (current_model_index + 1 >= model_names_to_load.Count)
            {
                SaveExperiment();
            }
            else
            { 
                current_model_index++;
                current_iteration = 0;
            }
        }
        // If not last iteration, then continue with the next iteration, which will be called on the next Update()
        else
        {
            current_iteration++;
        }
    }

    private void SaveExperiment()
    {
        Debug.Log("Done with calculations. Saving the Experiment to .json.");
        

        string filePath = "Assets/Experiments/BackwardsPassExperiment.json";
        string jsonData = JsonConvert.SerializeObject(experiments, Formatting.Indented);

        // Write the JSON data to the file
        File.WriteAllText(filePath, jsonData);

        dm.experiment_running = false;
        experiment_running = false;
    }

    private int FindIndexOfModelName()
    {
        return hc.models_available.IndexOf(model_names_to_load[current_model_index]);
    }

    private float CalculateTotalDistancesToEachother(List<Vector3> positions)
    {
        float totalDistance = 0;

        // Iterate through all positions in the List
        for (int position_index_1 = 0; position_index_1 < positions.Count-1; position_index_1++)
        {
            // Start with position_index_1 + 1 to not compare twice and not with itself.
            for (int position_index_2 = position_index_1+1; position_index_2 < positions.Count; position_index_2++)
            {
                // Add Distance between these two points to the total distance
                totalDistance += Vector3.Distance(positions[position_index_1], positions[position_index_2]);
            }
        }

        return totalDistance;
    }

    private List<List<Coordinate>> ConvertToCoordinates(List<List<Vector3>> vectorLists)
    {
        var coordinateLists = new List<List<Coordinate>>();

        foreach (var vectorList in vectorLists)
        {
            var coordinateList = new List<Coordinate>();

            foreach (var vector in vectorList)
            {
                var coordinate = new Coordinate
                {
                    x = vector.x,
                    y = vector.y,
                    z = vector.z
                };

                coordinateList.Add(coordinate);
            }

            coordinateLists.Add(coordinateList);
        }

        return coordinateLists;
    }

    // Class to save to the JSON File
    class Experiment
    {
        public string Name { get; set; }
        public List<List<Coordinate>> PositionsPerHiddenLayer { get; set; }
        public List<float> TotalDistancesToEachother { get; set; }

    }

    class Coordinate
    {
        public float x { get; set; }
        public float y { get; set; }
        public float z { get; set; }
    }*/
}
