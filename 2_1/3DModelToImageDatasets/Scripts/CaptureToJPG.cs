using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.UI;
using static UnityEngine.GraphicsBuffer;

public class CaptureToJPG : MonoBehaviour
{
    public Camera cameraToCapture;
    public Transform DirectionRight;
    private int imageWidth = 1920;
    private int imageHeight = 1080;
    private string savePath = "datasets\\LRF\\";
    private RenderTexture renderTexture;

    public GameObject LRF;
    public GameObject GLTD;
    public GameObject IRP;

    public Slider CameraY;
    public Slider FOV;
    public Slider LightAngleX;

    private void Awake()
    {
        CameraMoving.Target = LRF.transform;
        LRF.SetActive(true);
        GLTD.SetActive(false);
        IRP.SetActive(false);
    }
    // Start is called before the first frame update
    void Start()
    {
        renderTexture = new RenderTexture(imageWidth, imageHeight, 24);
    }

    IEnumerator CaptureRenderTextureToJPG()
    {
        while (true)
        {
            // 카메라 y축 랜덤 설정
            Vector3 position = cameraToCapture.transform.position;

            position.y = UnityEngine.Random.Range(0f, 0.5f);

            cameraToCapture.transform.position = position;

            // 라이트 각도 랜덤 설정
            float randomAngle = UnityEngine.Random.Range(10f, 360f);

            DirectionRight.transform.Rotate(randomAngle, 0, 0);

            // fov 랜덤 설정
            cameraToCapture.fieldOfView = UnityEngine.Random.Range(30f, 60f);

            // 카메라 배경 색상 랜덤 설정
            Color randomColor = new Color(UnityEngine.Random.value, UnityEngine.Random.value, UnityEngine.Random.value, 1.0f);

            cameraToCapture.backgroundColor = randomColor;

            yield return null;

            renderTexture = new RenderTexture(imageWidth, imageHeight, 24);
            DateTime now = DateTime.Now;

            // MM_dd_hh:mm:ss 형식으로 포맷팅
            string formattedDate = now.ToString("MM_dd_HH_mm_ss");

            cameraToCapture.targetTexture = renderTexture;
            // 카메라로 렌더링
            cameraToCapture.Render();

            yield return new WaitForEndOfFrame();

            RenderTexture.active = renderTexture;
            Texture2D texture2D = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);
            texture2D.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0);
            texture2D.Apply();

            // RenderTexture 해제
            cameraToCapture.targetTexture = null;
            RenderTexture.active = null;
            Destroy(renderTexture);

            // Texture2D를 JPG로 인코딩 및 저장
            byte[] bytes = texture2D.EncodeToJPG();
            string filePath = Path.Combine(savePath + formattedDate + ".jpg");
            File.WriteAllBytes(filePath, bytes);
            yield return new WaitForEndOfFrame();
            // Texture2D 해제
            Destroy(texture2D);

            Debug.Log("이미지가 저장되었습니다: " + formattedDate);

            yield return new WaitForSecondsRealtime(1);
        }
    }

    public void LRFButton()
    {
        savePath = "datasets\\LRF\\";

        CameraMoving.Target = LRF.transform;

        LRF.SetActive(true);
        GLTD.SetActive(false);
        IRP.SetActive(false);
    }

    public void GLTDButton()
    {
        savePath = "datasets\\GLTD\\";

        CameraMoving.Target = GLTD.transform;

        LRF.SetActive(false);
        GLTD.SetActive(true);
        IRP.SetActive(false);
    }

    public void IRPButton()
    {
        savePath = "datasets\\IRP\\";

        CameraMoving.Target = IRP.transform;

        LRF.SetActive(false);
        GLTD.SetActive(false);
        IRP.SetActive(true);
    }

    public void CaptureStart()
    {
        StartCoroutine(CaptureRenderTextureToJPG());
    }

    public void CaptureStop()
    {
        StopCoroutine(CaptureRenderTextureToJPG());
    }

    public void CameraYValueChange()
    {
        cameraToCapture.transform.position = new Vector3(cameraToCapture.transform.position.x, CameraY.value, cameraToCapture.transform.position.z);
    }
    public void FOVValueChange()
    {
        cameraToCapture.fieldOfView = FOV.value;
    }
    public void LightRotateValueChange()
    {
        DirectionRight.rotation = Quaternion.Euler(LightAngleX.value, transform.rotation.eulerAngles.y, transform.rotation.eulerAngles.z);
    }
}
