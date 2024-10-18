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
        // ��ũ���� ĸó
        yield return HTTPController.CaptureImage = ScreenCapture.CaptureScreenshotAsTexture();

        // ��������� �ؽ�ó ����
        //HTTPController.CaptureImage = ResizeTexture(HTTPController.CaptureImage, 224, 224);
        isGenerate = true;
    }

    private Texture2D ResizeTexture(Texture2D source, int newWidth, int newHeight)
    {
        // ���ο� �ؽ�ó ����
        Texture2D result = new Texture2D(newWidth, newHeight, source.format, false);

        // Resize �޼��带 ����Ͽ� ��������
        Graphics.ConvertTexture(source, result);

        return result;
    }
}
