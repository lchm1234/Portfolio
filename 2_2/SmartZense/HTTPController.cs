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
    // 서버 엔드포인트 URL
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

        // 이미지 파일 읽기
        Texture2D tex = CaptureImage;
        byte[] imageBytes = tex.EncodeToPNG();  // PNG로 인코딩하여 이미지 바이트 배열 생성

        // 이미지를 Base64로 인코딩
        string base64Image = Convert.ToBase64String(imageBytes);

        // JSON 데이터 생성
        ImageData imageData = new ImageData
        {
            image = base64Image
        };
        string jsonPayload = JsonUtility.ToJson(imageData);

        // POST 요청 생성
        UnityWebRequest request = new UnityWebRequest(serverURL, "POST");
        byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(jsonPayload);
        request.uploadHandler = new UploadHandlerRaw(bodyRaw);
        request.downloadHandler = new DownloadHandlerBuffer();
        request.SetRequestHeader("Content-Type", "application/json");

        // 요청 보내기
        yield return request.SendWebRequest();

        if (request.result != UnityWebRequest.Result.Success)
        {
            Debug.LogError($"Request failed: {request.error}");
        }
        else
        {
            // 응답 처리
            string jsonResponse = request.downloadHandler.text;

            // 예측 결과 처리 (예시: JSON 파싱)
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