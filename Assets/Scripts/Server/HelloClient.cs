using UnityEngine;
using Newtonsoft.Json;
using System.Collections.Generic;
using System;
using System.Text.RegularExpressions;
using Unity.VisualScripting;

public class HelloClient : MonoBehaviour
{
    private HelloRequester _helloRequester;

    public HelloRequester.task new_task = HelloRequester.task.nothing;
    public bool debug_trigger_task = false;
    public string input_train_or_test = "train";
    public int input_index = 0;
    public List<String> models_available;
    // Just for Debugging purposes, later just display the Elements of the List directly
    public int model_index = 0;

    [HideInInspector]
    public bool task_running = false;

    private DataManager usedDataManager = null;

    private void Start()
    {
        // Keep this, so that it only has to be loaded once
        usedDataManager = this.gameObject.GetComponent<DataManager>();

        _helloRequester = new HelloRequester();
        _helloRequester.task_to_do = HelloRequester.task.handshake;
        _helloRequester.Start();
        task_running = true;
    }

    private void Update()
    {
        // Logic for using inspector trigger to test out requests.
        // Can be deleted after all functionalities have been tested.
        if (debug_trigger_task)
        {
            if (task_running)
            {
                Debug.LogError("Another Task is already in progress!");
                debug_trigger_task = false;
                return;
            }
            if (new_task == HelloRequester.task.nothing)
            {
                Debug.LogError("No task set!");
                debug_trigger_task = false;
                return;
            }
            MakeANewRequest(new_task);
            debug_trigger_task = false;
        }
        // Debug Stuff ends here

        // Leave Update Loop if there is no Thread running
        if (!task_running) return;

        // Check if thread is done - If not, leave loop
        if (!_helloRequester.task_done) return;

        // Thread is done and task completed. Now work on the given task on main Thread.
        DealWithMessageAccordingToGivenTask();

        _helloRequester.Stop();
        _helloRequester = null;
        task_running = false;
    }

    private void OnDestroy()
    {
        _helloRequester.Stop();
    }

    /// <summary>
    /// Sends a new Request to the python server and deals with it's response accordingly.
    /// The sent request and response action is dependend on the new_task_to_do delivered.
    /// If the python server is already busy this medthod will refuse to to the task;
    /// </summary>
    /// <param name="new_task_to_do">The HelloRequester.task that is supposed to be accomplished.</param>
    public bool MakeANewRequest(HelloRequester.task new_task_to_do)
    {
        // If there is already a task that the python server is working on, don't except the new task!
        if (task_running) return false;

        _helloRequester = new HelloRequester();
        _helloRequester.task_to_do = new_task_to_do;
        SetParamsAccordingToGivenTask(new_task_to_do);
        _helloRequester.Start();
        task_running = true;

        return true;
    }

    private void SetParamsAccordingToGivenTask(HelloRequester.task new_task_to_do)
    {
        // Make sure that tasks that require aditional params are mentioned here
        switch (new_task_to_do)
        {
            case HelloRequester.task.load_input:
                _helloRequester.string_param_1 = input_train_or_test;
                _helloRequester.int_param_1 = input_index;
                break;
            case HelloRequester.task.load_and_send_input_and_activations:
                _helloRequester.string_param_1 = input_train_or_test;
                _helloRequester.int_param_1 = input_index;
                break;
            case HelloRequester.task.load_and_send_input_and_ig:
                _helloRequester.string_param_1 = input_train_or_test;
                _helloRequester.int_param_1 = input_index;
                break;
            case HelloRequester.task.send_subset_activations:
                _helloRequester.int_param_1 = 1;
                break;
            case HelloRequester.task.load_model:
                // TODO: Add Check if the model_index is out of bounds
                _helloRequester.string_param_1 = models_available[model_index];
                break;
        }
    }

    //TODO: Logic for each HelloRequester.task
    private void DealWithMessageAccordingToGivenTask()
    {
        // Make sure that you have the logic for each task_to_do in this switch case!
        switch (_helloRequester.task_to_do)
        {
            case HelloRequester.task.handshake:
                DealWithHandshake();
                break;
            case HelloRequester.task.load_simple_mlp:
                if (_helloRequester.messages is null)
                {
                    Debug.LogError("ERROR: The simple MLP model wasn't loaded.");
                }
                else
                {
                    Debug.Log("Simple MLP model loaded succesfully!");
                }
                break;
            case HelloRequester.task.load_model:
                if (_helloRequester.messages is null)
                {
                    Debug.LogError("ERROR: The " + models_available[model_index] + " model wasn't loaded.");
                }
                else
                {
                    Debug.Log(models_available[model_index] + " model loaded succesfully!");
                }
                break;
            case HelloRequester.task.load_and_send_input_and_activations:
                DealWithInputLoadingAndActivationDisplay();
                break;
            case HelloRequester.task.load_and_send_input_and_ig:
                DealWithInputLoadingAndIGDisplay();
                break;
            case HelloRequester.task.send_naps:
                DealWithNAPS();
                break;
            case HelloRequester.task.send_weighted_activations:
                DealWithWeightedActivations();
                break;
            case HelloRequester.task.send_average_activations:
                DealWithAverageActivations();
                break;
            case HelloRequester.task.send_subset_activations:
                DealWithSubsetActivations();
                break;
            case HelloRequester.task.send_average_signals:
                DealWithAverageSignals();
                break;
            case HelloRequester.task.display_weights:
                DealWithWeightDisplay();
                break;
            case HelloRequester.task.test_tensor:
                Debug.Log("Done");
                break;
            default:
                Debug.LogError("It seems like you are missing a task_to_do in the HelloClient.cs Function 'DealWithMessageAccordingToGiven Task()!");
                Debug.LogError(_helloRequester.task_to_do);
                break;
        }
    }

    private void DealWithHandshake()
    {
        if (_helloRequester.messages is null)
        {
            Debug.LogError("ERROR: No answer after attempted handshake recieved.");
            return;
        }

        List<string> res = TurnJSONintoStringList(_helloRequester.messages[0]);

        models_available = res;
    }

    private void DealWithInputLoadingAndActivationDisplay()
    {
        if (_helloRequester.messages is null)
        {
            Debug.LogError("ERROR: Input Loading and recieving Activations and Input failed.");
            return;
        }

        double[,] input = TurnJSONintoDoubleArray(_helloRequester.messages[0]);

        List<double[]> activations = new List<double[]>();

        for (int index = 1; index < _helloRequester.messages.Count; index++)
        {
            activations.Add(TurnJSONintoDoubleList(_helloRequester.messages[index]));
        }

        usedDataManager.InitParticleManagerForInput(input, activations);
    }

    private void DealWithInputLoadingAndIGDisplay()
    {
        if (_helloRequester.messages is null)
        {
            Debug.LogError("ERROR: Input Loading and recieving Activations and Input failed.");
            return;
        }

        double[,] input = TurnJSONintoDoubleArray(_helloRequester.messages[0]);

        double[,] ig = TurnJSONintoDoubleArray(_helloRequester.messages[1]);

        usedDataManager.InitSphereManager(input, ig);
    }

    private void DealWithNAPS()
    {
        if (_helloRequester.messages is null)
        {
            Debug.LogError("ERROR: The Server didn't respond to send the NAPs.");
            return;
        }

        if (_helloRequester.messages[0].Equals("No model set yet"))
        {
            Debug.LogError("ERROR: No model has been set yet. Please load a model first!");
            return;
        }

        List<double[,]> nap_arrays = new List<double[,]>();

        foreach (string message in _helloRequester.messages)
        {
            double[,] converted_message = TurnJSONintoDoubleArray(message);
            nap_arrays.Add(converted_message);
        }

        usedDataManager.InitParticleManagerForNaps(nap_arrays, 0);
    }

    private void DealWithWeightDisplay()
    {
        if(_helloRequester.messages is null)
        {
            Debug.LogError("ERROR: The Server didn't respond to send the Weights.");
            return;
        }
        if (_helloRequester.messages[0].Equals("No model set yet"))
        {
            Debug.LogError("ERROR: No model has been set yet. Please load a model first!");
            return;
        }
        List<double[,]> weight_arrays = new List<double[,]>();
        List<double[]> biases = new List<double[]>();
        for (int index = 0; index < _helloRequester.messages.Count/2; index++)
        {
            double[,] converted_message_weight = TurnJSONintoDoubleArray(_helloRequester.messages[index*2]);
            double[] converted_message_bias = TurnJSONintoDoubleList(_helloRequester.messages[index * 2 + 1]);
            weight_arrays.Add(converted_message_weight);
            biases.Add(converted_message_bias);
        }

        /*foreach (string message in _helloRequester.messages)
        {
            double[,] converted_message_weight= TurnJSONintoDoubleArray(message);
            double[] converted_message_bias = TurnJSONintoDoubleList(message);
            weight_arrays.Add(converted_message_weight);
            biases.Add(converted_message_bias);
        }*/

        usedDataManager.InitParticleManagerMLP(weight_arrays, biases);

    }

    private void DealWithWeightedActivations()
    {
        if (_helloRequester.messages is null)
        {
            Debug.LogError("ERROR: The Server didn't respond to send the Weights.");
            return;
        }
        if (_helloRequester.messages[0].Equals("No model set yet"))
        {
            Debug.LogError("ERROR: No model has been set yet. Please load a model first!");
            return;
        }

        List<double[,]> converted_weighted_activations = new List<double[,]>();
        foreach (String message in _helloRequester.messages)
        {
            converted_weighted_activations.Add(TurnJSONintoDoubleArray(message));
        }

        usedDataManager.InitWeightedActivationLines(converted_weighted_activations);
    }

    private void DealWithAverageActivations()
    {
        if (_helloRequester.messages is null)
        { 
            Debug.LogError("ERROR: The Server didn't respond to send the Weights.");
            return;
        }

        List<double[]> average_activations = new List<double[]>();

        foreach (String message in _helloRequester.messages)
        {
            average_activations.Add(TurnJSONintoDoubleList(message));
        }

        usedDataManager.InitClusteringManager(average_activations);
        
    }

    private void DealWithSubsetActivations()
    {
        if (_helloRequester.messages is null)
        {
            Debug.LogError("ERROR: The Server didn't respond to send the Weights.");
            return;
        }

        List<float[][]> subset_activations = new List<float[][]>();

        foreach (String message in _helloRequester.messages)
        {
            subset_activations.Add(TurnJSONIntoNestedDoubleArray(message));
        }

        usedDataManager.InitUmapLayout(subset_activations);
    }

    private void DealWithAverageSignals()
    {
        if (_helloRequester.messages is null)
        {
            Debug.LogError("ERROR: The Server didn't respond to send the Weights.");
            return;
        }

        List<float[][]> average_signals = new List<float[][]>();

        foreach (String message in _helloRequester.messages)
        {
            average_signals.Add(TurnJSONIntoNestedDoubleArray(message));
        }

        usedDataManager.InitUmapLayout(average_signals, true);
        // TODO: Remove Debug Stuff if not ever needed anymore
        /*foreach (float[][] array in average_signals)
        {
            Debug.Log(array.GetLength(0));
            Debug.Log(array[0].GetLength(0));
            Debug.Log("...");
        }

        float[][] centered_embeddings = new UmapReduction().applyUMAP(average_signals[1]);

        int tmp_count = 0;
        foreach (float[] coordinates in centered_embeddings)
        {
            Debug.Log("Coordinates of Neuron " + tmp_count.ToString() + ": " +
                "(" + coordinates[0].ToString() + "," + coordinates[1].ToString());
            tmp_count++;
        }*/
    }

    // TODO: Turn this Regex mess into actually using Newtonsoft.JSON.. Didn't work initially 
    private List<string> TurnJSONintoStringList(string message)
    {
        Debug.Log("Message looks like:");
        Debug.Log(message);
        try
        {
            List<string> response = new List<string>();
            MatchCollection matches = Regex.Matches(message, @"\\\""(.*?)\\\""");

            // Extract the matched strings and add them to the options list
            foreach (Match match in matches)
            {
                response.Add(match.Groups[1].Value);
            }
            
            return response;
        }
        catch (JsonException)
        {
            Debug.LogError("Error deserializing JSON string.");
            return null;
        }
    }

    private float[,] TurnJSONintoFloatArray(string message)
    {
        try
        {
            string trimmed_message = message.Trim('"');
            // Deserialize the JSON string into a float array
            float[,] twoDimensionalArray = JsonConvert.DeserializeObject<float[,]>(trimmed_message);
            return twoDimensionalArray;
        }
        catch (JsonException)
        {
            Debug.LogError("Error deserializing JSON string.");
            return null;
        }
    }

    private double[,] TurnJSONintoDoubleArray(string message)
    {
        try
        {
            string trimmed_message = message.Trim('"');
            // Deserialize the JSON string into a float array
            double[,] twoDimensionalArray = JsonConvert.DeserializeObject<double[,]>(trimmed_message);
            return twoDimensionalArray;
        }
        catch (JsonException)
        {
            Debug.LogError("Error deserializing JSON string.");
            Debug.Log(message);
            return null;
        }
    }

    private double[] TurnJSONintoDoubleList(string message)
    {
        try
        {
            string trimmed_message = message.Trim('"');
            // Deserialize the JSON string into a float array
            double[] twoDimensionalArray = JsonConvert.DeserializeObject<double[]>(trimmed_message);
            return twoDimensionalArray;
        }
        catch (JsonException)
        {
            Debug.LogError("Error deserializing JSON string.");
            Debug.Log(message);
            return null;
        }
    }

    private float[][] TurnJSONIntoNestedDoubleArray(string message)
    {
        try
        {
            string trimmed_message = message.Trim('"');
            // Deserialize the JSON string into a float array
            float[][] nestedArray = JsonConvert.DeserializeObject<float[][]>(trimmed_message);
            return nestedArray;
        }
        catch (JsonException)
        {
            Debug.LogError("Error deserializing JSON string.");
            Debug.Log(message);
            return null;
        }
    }
}
