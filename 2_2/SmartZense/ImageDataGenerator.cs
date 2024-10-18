using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.UI;

public class ImageDataGenerator : MonoBehaviour
{
    public static ImageDataGenerator Instance;
    public bool isGenerate;
    private void Awake()
    {
        Instance = this;
    }

    public void GenerateImageData()
    {
        StartCoroutine(CaptureAndLoadScreenshot());
    }

    private IEnumerator CaptureAndLoadScreenshot()
    {
        // 스크린샷 캡처
        yield return HTTPController.CaptureImage = ScreenCapture.CaptureScreenshotAsTexture();

        // 리사이즈된 텍스처 생성
        //HTTPController.CaptureImage = ResizeTexture(HTTPController.CaptureImage, 224, 224);
        isGenerate = true;
    }

    private Texture2D ResizeTexture(Texture2D source, int newWidth, int newHeight)
    {
        // 새로운 텍스처 생성
        Texture2D result = new Texture2D(newWidth, newHeight, source.format, false);

        // Resize 메서드를 사용하여 리사이즈
        Graphics.ConvertTexture(source, result);

        return result;
    }
}
