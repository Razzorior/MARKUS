using System.Collections;
using UnityEngine;
using System.Collections.Generic;
using System;
using System.Linq;

public class ClusterTree : MonoBehaviour
{
    Dictionary<int, ValueContainer> dict = new Dictionary<int, ValueContainer>();
    public Node root = null;
    public int max_size;

    public ClusterTree(double[] attribute)
    {
        max_size = attribute.Length;

        for (int index = 0; index < max_size; index++)
        {
            dict.Add(index, new ValueContainer { SingleDouble = (attribute[index]) });
        }
        
        CreateTree(attribute);
    }

    public ClusterTree(double[,] attribute)
    {
        max_size = attribute.GetLength(0);

        for (int index = 0; index < max_size; index++)
        {
            dict.Add(index, new ValueContainer { ArrayDouble = GetRow(attribute, index) });
        }

        CreateTree(attribute);
    }

    public List<Node> GetClusters(int amount_of_clusters)
    {
        if (amount_of_clusters > max_size)
        {
            throw new ArgumentException("Too many clusters requested. Only " + max_size.ToString() + " max clusters possible, however "
                + amount_of_clusters.ToString() + " where requested"); 
        }

        List<Node> result = new List<Node>(amount_of_clusters+1);
        result.Add(root);

        for (int depth = 1; depth < amount_of_clusters; depth++)
        {
            // Find next cluster to unpack 
            double max_distance = double.MinValue;
            int max_dist_index = 0;
            for (int node_index = 0; node_index < result.Count; node_index++)
            {
                // Skip if leaf node
                if (result[node_index].children.Count == 0) { continue; }

                if (result[node_index].length > max_distance)
                {
                    max_distance = result[node_index].length;
                    max_dist_index = node_index;
                }
            }

            // Distance has to be positive. Negative must be an error in calculation, 0 means there is only leaves left. 
            if (max_distance < 0)
            {
                throw new InvalidOperationException("There must be an error with the tree calculation, " +
                    "as further unpacking of the tree is not possible. Distance is: " + max_distance.ToString());
            }

            // Add children to List, to unpack the cluster
            result.Add(result[max_dist_index].children[0]);
            result.Add(result[max_dist_index].children[1]);

            // Remove unpacked cluster
            result.RemoveAt(max_dist_index);
        }

        return result;
    }

    private void CreateTree(double[] values)
    {
        List<Node> active_clusters = new List<Node>();
        // Add initial leaves
        for (int index = 0; index < values.Length; index++)
        {
            active_clusters.Add(new Node(dict, null, new List<Node>(0), new List<int> { index }, 0.0, false));
        }


        double[,] distance_matrix = InitDistanceMatrix(active_clusters);

        // Do until there is only one (root) cluster left
        while (active_clusters.Count > 1)
        {
            (int index1, int index2, double distance) = FindClosestDistance(distance_matrix, active_clusters.Count);
            UpdateDistanceMatrix(index1, index2, distance, ref active_clusters, ref distance_matrix);
        }

        root = active_clusters[0];
    }

    // TODO: Edit this function
    private void CreateTree(double[,] values)
    {
        List<Node> active_clusters = new List<Node>();
        // Add initial leaves
        for (int index = 0; index < values.GetLength(0); index++)
        {
            active_clusters.Add(new Node(dict, null, new List<Node>(0), new List<int> { index }, 0.0, true));
        }

        //Debug.Log("Initial Leaves Count: " + active_clusters.Count);
        double[,] distance_matrix = InitDistanceMatrix(active_clusters);

        int test = 0;
        // Do until there is only one (root) cluster left
        while (active_clusters.Count > 1)
        {
            //Debug.Log("Going along the tree in step  " + test.ToString());
            (int index1, int index2, double distance) = FindClosestDistance(distance_matrix, active_clusters.Count);
            //Debug.Log("Index " + index1.ToString() + " and Index " + index2.ToString() + " are closest with the distance: " + distance.ToString());
            if (distance == 0.0)
            {
                String str = "Average array of first array: ";
                foreach (double element in active_clusters[index1].average_array)
                {
                    str += (element.ToString() + " ");
                }
                //Debug.Log(str);

                str = "Average array of second array: ";
                foreach (double element2 in active_clusters[index2].average_array)
                {
                    str += (element2.ToString() + " ");
                }
                //Debug.Log(str);
            }
            UpdateDistanceMatrix(index1, index2, distance, ref active_clusters, ref distance_matrix);
            test++;
        }
        //Debug.Log(active_clusters[0].children.Count);

        root = active_clusters[0];

    }

    private double[,] InitDistanceMatrix(List<Node> clusters)
    {
        double[,] result = new double[clusters.Count, clusters.Count];

        for (int i = 0; i < clusters.Count; i++)
        {
            // Distance of cluster to itself is zero
            result[i, i] = 0.0;

            // Iterate through all distances 
            for (int j = 0; j < i; j++)
            {
                double dist = clusters[i].Distance(clusters[j]);
                result[i, j] = dist;
                result[j, i] = dist;
            }
        }

        return result;
    }

    private (int, int, double) FindClosestDistance(double[,] dist_mat, int actual_mat_size)
    {
        if (dist_mat.GetLength(0) != dist_mat.GetLength(1))
        {
            throw new ArgumentException("Distance Matrix is not symmetrical");
        }

        if (dist_mat.GetLength(0) <= 1)
        {
            throw new ArgumentException("Distance Matrix is of size 1. Clustering Algorithm needs to stop at the root.");
        }

        int index1 = 0;
        int index2 = 1;
        double smallest_dist = dist_mat[index1, index2];

        for (int i = 0; i < actual_mat_size; i++)
        {
            // j < i ensures not considering cluster distance to istelf, which is always 0. 
            for (int j = 0; j < i; j++)
            {
                if (smallest_dist > dist_mat[i, j])
                {
                    index1 = i;
                    index2 = j;
                    smallest_dist = dist_mat[i, j];
                }
            }
        }
        return (index1, index2, smallest_dist);
    }

    private void UpdateDistanceMatrix(int index1, int index2, double distance, ref List<Node> active_clusters, ref double[,] dist_mat)
    {
        //double[,] new_dist_mat = new double[old_dist_mat.GetLength(0) - 1, old_dist_mat.GetLength(1) - 1];

        // Add new merged cluster to list and remove it's old individual clusters from list
        List<int> merged_neuron_ids = active_clusters[index1].neuron_ids.Concat(active_clusters[index2].neuron_ids).ToList();
        Node new_cluster = new(dict, null, new List<Node> { active_clusters[index1], active_clusters[index2] }, merged_neuron_ids, distance / 2.0, active_clusters[index1].is_array);
        //Debug.Log(active_clusters[index1]);
        //Debug.Log(active_clusters[index2]);
        active_clusters[index1].parent = new_cluster;
        active_clusters[index2].parent = new_cluster;

        active_clusters.Add(new_cluster);

        if (index1 > index2)
        {
            active_clusters.RemoveAt(index1);
            active_clusters.RemoveAt(index2);
        }
        else
        {
            active_clusters.RemoveAt(index2);
            active_clusters.RemoveAt(index1);
        }

        

        // Shifting elements in distance matrix, leaving out the merged elements. i<= because active_clusters is already smaller than old matrix
        int offset_i = 0;
        for (int i = 0; i <= active_clusters.Count; i++) 
        {
            if (i == index1 || i == index2)
            {
                offset_i++;
                continue;
            }

            int offset_j = 0;
            for (int j = 0; j <= active_clusters.Count; j++)
            {
                if (j == index1 || j == index2)
                {
                    offset_j++;
                    continue;
                }
                // Values haven't changed, so nothing needs to be shifted. 
                if (offset_i == 0 && offset_j == 0)
                {
                    continue;
                }

                dist_mat[i - offset_i, j - offset_j] = dist_mat[i, j];
            }
        }

        // Calculate distances of all clusters to the new merged clusters and save them into the distance matrix.
        int mat_len = active_clusters.Count;
        for (int i = 0; i < mat_len-1; i++)
        {
            double dist = new_cluster.Distance(active_clusters[i]);
            dist_mat[i, mat_len] = dist;
            dist_mat[mat_len, i] = dist;
        }
        dist_mat[mat_len, mat_len] = 0.0;

    }

    private double[] GetRow(double[,] matrix, int rowIndex)
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

public class Node
{
    public Node parent { get; set; }
    public List<Node> children { get; set; }
    public List<int> neuron_ids { get; set; }
    public bool is_array { get; set; }
    public double length { get; set; }
    public double average_single { get; set; }
    public double[] average_array { get; set; }

    public Node(Dictionary<int, ValueContainer> dict, Node _parent, List<Node> _child, List<int> _neuron_ids, double _length, bool _is_array)
    {
        parent = _parent;
        children = _child;
        neuron_ids = _neuron_ids;
        length = _length;
        is_array = _is_array;

        if (is_array)
        {
            double[][] arraysToAverage = neuron_ids.ToArray()
            .Where(id => dict.ContainsKey(id))
            .Select(id => dict[id].ArrayDouble)
            .ToArray();

            // If leaf node, no averaging is required
            if (_neuron_ids.Count == 1)
            {
                average_array = arraysToAverage[0];
            }
            /*
            foreach (double[] debora in arraysToAverage)
            {
                String str = "new row: ";
                foreach (double element in debora)
                {
                    str += (element.ToString() + " ");
                }
                Debug.Log(str);
            }
            Debug.Log("Done wit it");
            */
            else
            {
                average_array = AverageArrays(arraysToAverage);
            }
        }
        else
        {
            double[] arrayToAverage = neuron_ids.ToArray()
            .Where(id => dict.ContainsKey(id))
            .Select(id => dict[id].SingleDouble)
            .ToArray();

            if (_neuron_ids.Count == 1)
            {
                average_single = arrayToAverage[0];
            }
            else
            {
                average_single = arrayToAverage.Average();
            }
        }

        static double[] AverageArrays(params double[][] arrays)
        {
            if (arrays.Length == 0)
            {
                throw new ArgumentException("At least one array must be provided.");
            }

            int arrayLength = arrays[0].Length;

            // Check that all arrays have the same length
            if (!arrays.All(arr => arr.Length == arrayLength))
            {
                throw new ArgumentException("All arrays must have the same length.");
            }

            double[] averageArray = new double[arrayLength];

            // Compute the average element-wise
            for (int i = 0; i < arrayLength; i++)
            {
                averageArray[i] = arrays.Select(arr => arr[i]).Average();
            }

            return averageArray;
        }
    }

    public double Distance(Node other_node)
    {
        if (is_array)
        {
            return average_array.Zip(other_node.average_array, (x, y) => Math.Abs(x - y)).Sum();
        }
        else
        {
            return Math.Abs(average_single - other_node.average_single);
        }
    }
}

public class ValueContainer
{
    public double SingleDouble { get; set; }
    public double[] ArrayDouble { get; set; }
}