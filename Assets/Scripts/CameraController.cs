using UnityEngine;

public class CameraController : MonoBehaviour
{
    public float speed = 20.0f;
    public float rotationSpeed = 100.0f;
    public float mouseSensitivity = 2.0f;

    private float pitch = 0.0f;
    private float yaw = 0.0f;

    private Rigidbody rb;

    void Start()
    {
        rb = GetComponent<Rigidbody>();
        rb.useGravity = false; // Disable gravity
        rb.drag = 0; // Ensure drag is set to 0
        rb.angularDrag = 0; // Ensure angular drag is set to 0
        rb.constraints = RigidbodyConstraints.FreezeRotationX | RigidbodyConstraints.FreezeRotationY  |RigidbodyConstraints.FreezeRotationZ;
    }

    void Update()
    {
        // Movement
        float translation = Input.GetAxis("Vertical") * speed * Time.deltaTime;
        float strafe = Input.GetAxis("Horizontal") * speed * Time.deltaTime;
        float vertical = 0.0f;

        if (Input.GetKey(KeyCode.LeftShift) || Input.GetKey(KeyCode.RightShift))
        {
            vertical = speed * Time.deltaTime; // Move up
        }
        if (Input.GetKey(KeyCode.LeftControl) || Input.GetKey(KeyCode.RightControl))
        {
            vertical = -speed * Time.deltaTime; // Move down
        }

        Vector3 movement = new Vector3(strafe, vertical, translation);
        transform.Translate(movement, Space.Self);

        // Rotation with keys
        if (Input.GetKey(KeyCode.Q))
        {
            transform.Rotate(Vector3.up, -rotationSpeed * Time.deltaTime);
        }
        if (Input.GetKey(KeyCode.E))
        {
            transform.Rotate(Vector3.up, rotationSpeed * Time.deltaTime);
        }

        // Rotation with mouse when right mouse button is held
        if (Input.GetMouseButton(1)) // Right mouse button
        {
            yaw += Input.GetAxis("Mouse X") * mouseSensitivity;
            pitch -= Input.GetAxis("Mouse Y") * mouseSensitivity;
            pitch = Mathf.Clamp(pitch, -90f, 90f); // Limit pitch to avoid flipping

            transform.eulerAngles = new Vector3(pitch, yaw, 0.0f);
        }
    }

    void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.CompareTag("Wall"))
        {
            // Apply force to push away from the wall
            rb.AddForce(collision.contacts[0].normal * -100f, ForceMode.Impulse);

            // Reduce velocity
            rb.velocity *= 0.5f;
        }
    }
}