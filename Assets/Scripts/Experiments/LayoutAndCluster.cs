using System.Collections;
using System.Collections.Generic;
using UnityEngine;

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
            }
        }
    }


}
