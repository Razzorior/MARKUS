using AsyncIO;
using NetMQ;
using NetMQ.Sockets;
using UnityEngine;
using Newtonsoft.Json;
using System.Collections.Generic;
using System;
using UnityEngine.Networking.PlayerConnection;

// TODO: Write multiple classes for each task I guess.. Maybe come up with a better solution
public class HelloRequester : RunAbleThread
{
    public enum task
    {
        handshake,
        load_model,
        load_input,
        load_and_send_input_and_activations,
        load_and_send_input_and_ig,
        load_model_and_send_class_analysis_data,
        large_mlp,
        cnn,
        test,
        test_tensor,
        send_naps,
        send_input,
        send_weighted_activations,
        display_weights,
        send_average_activations,
        send_class_average_activations,
        send_subset_activations,
        send_average_signals,
        send_class_predictions_activations_and_sigs,
        send_class_average_signals,
        send_class_analysis_data,
        nothing
    }

    public task task_to_do = task.nothing;
    public String string_param_1 = null;
    public int int_param_1;
    public DataManager.UMAPData umapdata;
    public DataManager dataManager = null;
    public List<String> messages = null;

    protected override void Run()
    {
        // Exit function if there is no known task to be done
        if (task_to_do == task.nothing) { return; }

        ForceDotNet.Force(); // this line is needed to prevent unity freeze after one use, not sure why yet
        using (RequestSocket client = new RequestSocket())
        {
            client.Connect("tcp://localhost:5555");
            switch (task_to_do)
            {
                case task.handshake:
                    Handshake(client);
                    break;
                case task.load_input:
                    LoadInput(client);
                    break;
                case task.load_model:
                    LoadModel(client);
                    break;
                case task.load_and_send_input_and_activations:
                    LoadAndSendInputAndActivations(client);
                    break;
                case task.load_and_send_input_and_ig:
                    LoadAndSendInputAndIG(client);
                    break;
                case task.load_model_and_send_class_analysis_data:
                    LoadModelAndSendClassAnalysisData(client);
                    break;
                case task.test:
                    ExecuteTestMessages(client);
                    break;
                case task.test_tensor:
                    TestTensor(client);
                    break;
                case task.display_weights:
                    DisplayWeights(client);
                    break;
                case task.send_naps:
                    SendNAPs(client);
                    break;
                case task.send_weighted_activations:
                    SendWeightedActivations(client);
                    break;
                case task.send_average_activations:
                    SendAverageActivaions(client);
                    break;
                case task.send_class_average_activations:
                    SendClassAverageActivations(client);
                    break;
                case task.send_subset_activations:
                    SendSubsetActivations(client);
                    break;
                case task.send_average_signals:
                    SendAverageSignals(client);
                    break;
                case task.send_class_average_signals:
                    SendClassAverageSignals(client);
                    break;
                case task.send_class_predictions_activations_and_sigs:
                    SendClassPredictionsActivationsAndSigs(client);
                    break;
                case task.send_class_analysis_data:
                    SendClassAnalysisData(client);
                    break;
            }
        }
        NetMQConfig.Cleanup(); // this line is needed to prevent unity freeze after one use
        task_done = true;
    }

    private void DisplayWeights(RequestSocket client)
    {
        client.SendFrame("give_weights");
        bool success = false;
        List<String> message = new List<string>();

        while (Running)
        {
            success = client.TryReceiveMultipartStrings(TimeSpan.FromSeconds(2), ref message);
            if (success) break;
        }

        messages = message;
        return;
    }
    private void ExecuteTestMessages(RequestSocket client)
    {
        for (int i = 0; i < 10 && Running; i++)
        {
            Debug.Log("Sending Hello");
            client.SendFrame("text");
            string message = null;
            bool gotMessage = false;
            while (Running)
            {
                gotMessage = client.TryReceiveFrameString(out message);
                if (gotMessage) break;
            }

            if (gotMessage) Debug.Log("Received " + message);
        }
    }

    private void Handshake(RequestSocket client)
    {
        Debug.Log("Initiating Handshake");
        client.SendFrame("handshake");
        string message = null;
        bool gotMessage = false;
        while (Running)
        {
            gotMessage = client.TryReceiveFrameString(out message);
            if (gotMessage) break;
        }

        List<string> response = new List<string> { message };
        messages = response;
    }

    private void LoadInput(RequestSocket client)
    {
        client.SendFrame("load_input");
        bool success = false;
        string affirmation = "";

        while (Running)
        {
            success = client.TryReceiveFrameString(out affirmation);
            if (success) break;
        }

        Debug.Log("Recieved message: " + affirmation);

        // Now sending whether train or test set or any other specific set/class names
        success = false;
        affirmation = "";
        client.SendFrame(string_param_1);
        while (Running)
        {
            success = client.TryReceiveFrameString(out affirmation);
            if (success) break;
        }

        Debug.Log("Recieved message: " + affirmation);

        success = false;
        affirmation = "";
        client.SendFrame(int_param_1.ToString());

        while (Running)
        {
            success = client.TryReceiveFrameString(out affirmation);
            if (success) break;
        }

        Debug.Log("Recieved message: " + affirmation);

    }

    private void LoadModel(RequestSocket client)
    {
        client.SendFrame("load_model");
        bool success = false;
        string affirmation = "";

        while (Running)
        {
            success = client.TryReceiveFrameString(out affirmation);
            if (success) break;
        }

        Debug.Log("Recieved message: " + affirmation);

        // Now sending the name of the model
        success = false;
        affirmation = "";
        client.SendFrame(string_param_1);
        while (Running)
        {
            success = client.TryReceiveFrameString(out affirmation);
            if (success) break;
        }

        messages = new List<string> { affirmation };

        Debug.Log("Recieved message: " + affirmation);
    }

    private void LoadAndSendInputAndActivations(RequestSocket client)
    {
        LoadInput(client);
        messages = new List<string>();
        SendInput(client, false);
        SendActivations(client, false);
    }

    private void LoadAndSendInputAndIG(RequestSocket client)
    {
        LoadInput(client);
        messages = new List<string>();
        SendInput(client, false);
        SendIntegratedGradients(client, false);
    }

    private void LoadModelAndSendClassAnalysisData(RequestSocket client)
    {
        LoadModel(client);
        SendClassAnalysisData(client);
    }

    private void SendClassAnalysisData(RequestSocket client)
    {
        messages = new List<string>();
        SendClassAverageActivations(client, false);
        SendSubsetActivations(client, false);
        SendClassAverageSignals(client, false);
        SendClassPredictionsActivationsAndSigs(client, false);
    }

    private void TestTensor(RequestSocket client)
    {
        Debug.Log("Asking for Tensor of loaded model.");
        client.SendFrame("test_tensor");
        bool success = false;
        List<String> message = new List<string>();

        while (Running)
        {
            success = client.TryReceiveMultipartStrings(TimeSpan.FromSeconds(2), ref message);
            if (success) break;
        }

        Debug.Log("Received " + message.Count + " messages.");

        messages = message;
    }

    private void SendActivations(RequestSocket client, bool create_list = true)
    {
        client.SendFrame("send_activations");
        bool success = false;
        var message = new List<string>();
        while (Running)
        {
            success = client.TryReceiveMultipartStrings(TimeSpan.FromSeconds(2), ref message);
            if (success) break;
        }

        Debug.Log("Received " + message.Count + " messages.");

        if (create_list)
        {
            messages = message;
        }
        else
        {
            foreach (string mes in message)
            {
                messages.Add(mes);
            }
        }
    }
    /// <summary>
    /// Sends two Requests to the server. First one let's it know, that it wants the NAPs of the loaded model.
    /// Second Request sends the layers that it wants to have the NAPs of.
    /// The resulting JSON String is then converted to a list of float arrays and sent to initiate the Particle Manager,
    /// that then spawns the particles representing the NAPs.
    /// </summary>
    /// <param name="client"></param>
    private void SendNAPs(RequestSocket client)
    {
        Debug.Log("Asking the server to send NAPs of loaded model");
        client.SendFrame("send_naps");
        bool success = false;
        string first_response = null;
        while (Running)
        {
            success = client.TryReceiveFrameString(out first_response);

            if (success) break;
        }

        Debug.Log("Message received: " + first_response);

        var message = new List<string>();
        int[] layers = { 1, 2 };
        string json = JsonConvert.SerializeObject(layers);
        Debug.Log("Sending list of layers to server");
        client.SendFrame(json);
        success = false;
        while (Running)
        {
            success = client.TryReceiveMultipartStrings(TimeSpan.FromSeconds(2), ref message);

            if (success) break;

        }
        Debug.Log("Received " + message.Count + " messages.");

        messages = message;

        return;
    }

    private void SendInput(RequestSocket client, bool create_list = true)
    {
        client.SendFrame("send_input");
        bool success = false;
        string response = null;
        while (Running)
        {
            success = client.TryReceiveFrameString(out response);

            if (success) break;
        }

        if (create_list)
        {
            messages = new List<String> { response };
        }
        else
        {
            messages.Add(response);
        }

    }

    private void SendIntegratedGradients(RequestSocket client, bool create_list = true)
    {
        client.SendFrame("integrated_gradient");
        bool success = false;
        string response = null;
        while (Running)
        {
            success = client.TryReceiveFrameString(out response);

            if (success) break;
        }

        if (create_list)
        {
            messages = new List<String> { response };
        }
        else
        {
            messages.Add(response);
        }

    }

    private void SendWeightedActivations(RequestSocket client)
    {
        client.SendFrame("send_weighted_activations");
        bool success = false;
        List<String> message = new List<string>();

        while (Running)
        {
            success = client.TryReceiveMultipartStrings(TimeSpan.FromSeconds(2), ref message);
            if (success) break;
        }

        messages = message;
    }

    private void SendAverageActivaions(RequestSocket client)
    {
        client.SendFrame("send_average_activations");
        List<String> message = new List<string>();
        bool success = false;

        while (Running)
        {
            success = client.TryReceiveMultipartStrings(TimeSpan.FromSeconds(2), ref message);

            if (success) break;

        }
        messages = message;
    }

    private void SendClassAverageActivations(RequestSocket client, bool create_list = true)
    {
        client.SendFrame("send_class_average_activations");

        string affirmation = "";
        bool success = false;

        while (Running)
        {
            success = client.TryReceiveFrameString(out affirmation);
            if (success) break;
        }

        Debug.Log(affirmation);

        List<string> message = new List<string>();

        client.SendFrame(((int)umapdata).ToString());


        while (Running)
        {
            success = client.TryReceiveMultipartStrings(TimeSpan.FromSeconds(60), ref message);
            if (success) break;
        }

        Debug.Log("Received " + message.Count + " messages.");

        if (create_list)
        {
            messages = message;
        }
        else
        {
            foreach (string mes in message)
            {
                messages.Add(mes);
            }
        }

    }

    private void SendAverageSignals(RequestSocket client)
    {
        client.SendFrame("send_average_signals");
        List<String> message = new List<string>();
        bool success = false;

        while (Running)
        {
            success = client.TryReceiveMultipartStrings(TimeSpan.FromSeconds(2), ref message);

            if (success) break;

        }
        messages = message;
    }

    private void SendClassAverageSignals(RequestSocket client, bool create_list = true)
    {
        client.SendFrame("send_class_average_signals");
        string response = null;
        bool success = false;

        while (Running)
        {
            success = client.TryReceiveFrameString(out response);

            if (success) break;
        }

        if (create_list)
        {
            messages = new List<String> { response };
        }
        else
        {
            messages.Add(response);
        }
    }

    private void SendClassPredictionsActivationsAndSigs(RequestSocket client, bool create_list = true)
    {
        client.SendFrame("send_class_predictions_activations_and_sigs");
        List<String> message = new List<string>();
        bool success = false;

        while (Running)
        {
            success = client.TryReceiveMultipartStrings(TimeSpan.FromSeconds(2), ref message);

            if (success) break;
        }

        if (create_list)
        {
            messages = message;
        }
        else
        {
            foreach (string msg in message)
            {
                messages.Add(msg);
            }
        }
    }

    private void SendSubsetActivations(RequestSocket client, bool create_list = true)
    {
        client.SendFrame("send_subset_activations");
        bool success = false;
        string affirmation = "";

        while (Running)
        {
            success = client.TryReceiveFrameString(out affirmation);
            if (success) break;
        }

        Debug.Log("Recieved message: " + affirmation);
        List<String> message = new List<string>();
        client.SendFrame(int_param_1.ToString());

        string response = null;
        success = false;

        while (Running)
        {
            success = client.TryReceiveFrameString(out response);

            if (success) break;
        }

        if (create_list)
        {
            messages = new List<String> { response };
        }
        else
        {
            messages.Add(response);
        }
    }
 }