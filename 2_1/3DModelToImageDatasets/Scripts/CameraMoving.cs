using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CameraMoving : MonoBehaviour
{
    public static Transform Target;
    public float rotationSpeed = 10.0f;
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        transform.RotateAround(Target.position, Vector3.up, rotationSpeed * Time.deltaTime);

        transform.LookAt(Target);
    }
}
