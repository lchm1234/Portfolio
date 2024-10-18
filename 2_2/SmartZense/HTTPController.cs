using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;

[Serializable]
public class ImageData
{
    public string image;
}

public class HTTPController : MonoBehaviour
{
    public static HTTPController Instance;
    // ���� ��������Ʈ URL
    private string serverURL = "";

    public static Texture2D CaptureImage = null;

    private void Awake()
    {
        Instance = this;
    }

    public void Classifier()
    {
        StartCoroutine(SendImageAndGetPrediction());
    }

    IEnumerator SendImageAndGetPrediction()
    {
        while (!ImageDataGenerator.Instance.isGenerate)
        {
            yield return new WaitForSeconds(0.1f);
        }

        // �̹��� ���� �б�
        Texture2D tex = CaptureImage;
        byte[] imageBytes = tex.EncodeToPNG();  // PNG�� ���ڵ��Ͽ� �̹��� ����Ʈ �迭 ����

        // �̹����� Base64�� ���ڵ�
        string base64Image = Convert.ToBase64String(imageBytes);

        // JSON ������ ����
        ImageData imageData = new ImageData
        {
            image = base64Image
        };
        string jsonPayload = JsonUtility.ToJson(imageData);

        // POST ��û ����
        UnityWebRequest request = new UnityWebRequest(serverURL, "POST");
        byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(jsonPayload);
        request.uploadHandler = new UploadHandlerRaw(bodyRaw);
        request.downloadHandler = new DownloadHandlerBuffer();
        request.SetRequestHeader("Content-Type", "application/json");

        // ��û ������
        yield return request.SendWebRequest();

        if (request.result != UnityWebRequest.Result.Success)
        {
            Debug.LogError($"Request failed: {request.error}");
        }
        else
        {
            // ���� ó��
            string jsonResponse = request.downloadHandler.text;

            // ���� ��� ó�� (����: JSON �Ľ�)
            try
            {
                SimpleJSON.JSONNode json = SimpleJSON.JSON.Parse(jsonResponse);
                int predictedClass = json["predicted_class"].AsInt;
                Debug.Log($"Predicted class: {predictedClass}");
                switch(predictedClass)
                {
                    case 0:
                        UIController.Instance.OnGLTDPanel();
                        break;
                    case 1:
                        UIController.Instance.OnIRPPanel();
                        break;
                    case 2:
                        UIController.Instance.OnLRFPanel();
                        break;
                }
            }
            catch (Exception e)
            {
                Debug.LogError($"Failed to parse JSON response: {e.Message}");
            }
        }
        ImageDataGenerator.Instance.isGenerate = false;
    }
}