#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>

#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {

    string fuente = "0";

    if (argc > 1) {
        fuente = argv[1];
    }

    VideoCapture cap;

    if (fuente == "0") {
        cap.open(0);
    } else {
        cap.open(fuente);
    }

    if (!cap.isOpened()) {
        cerr << "Error: no se pudo abrir la cámara o el video." << endl;
        return -1;
    }

    int dispositivosCUDA = cuda::getCudaEnabledDeviceCount();

    if (dispositivosCUDA <= 0) {
        cerr << "Error: OpenCV no detecta GPU con CUDA." << endl;
        cerr << "Verifica que OpenCV esté compilado con soporte CUDA." << endl;
        return -1;
    }

    cuda::setDevice(0);
    cuda::DeviceInfo deviceInfo(0);

    cout << "GPU detectada: " << deviceInfo.name() << endl;
    cout << "Iniciando prueba CPU vs GPU..." << endl;

    ofstream csv("resultados_parte_1c.csv");
    csv << "frame,cpu_ms,gpu_ms\n";

    Mat frame;
    Mat frameResize;

    const int anchoObjetivo = 640;

    int frameCount = 0;
    int framesValidos = 0;
    int warmup = 10;

    double totalCpuMs = 0.0;
    double totalGpuMs = 0.0;

    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));

    Ptr<cuda::Filter> filtroGaussianoGPU =
        cuda::createGaussianFilter(CV_8UC1, CV_8UC1, Size(5, 5), 1.5);

    Ptr<cuda::Filter> erosionGPU =
        cuda::createMorphologyFilter(MORPH_ERODE, CV_8UC1, kernel);

    Ptr<cuda::Filter> dilatacionGPU =
        cuda::createMorphologyFilter(MORPH_DILATE, CV_8UC1, kernel);

    Ptr<cuda::CannyEdgeDetector> cannyGPU =
        cuda::createCannyEdgeDetector(50.0, 150.0);

    cuda::Stream stream;

    while (true) {

        cap >> frame;

        if (frame.empty()) {
            break;
        }

        frameCount++;

        double escala = static_cast<double>(anchoObjetivo) / frame.cols;
        resize(frame, frameResize, Size(), escala, escala);

        // ==========================
        // PIPELINE CPU
        // ==========================
        Mat grayCPU, blurCPU, equalCPU, morphCPU, edgesCPU;

        auto inicioCPU = chrono::high_resolution_clock::now();

        cvtColor(frameResize, grayCPU, COLOR_BGR2GRAY);
        GaussianBlur(grayCPU, blurCPU, Size(5, 5), 1.5);
        equalizeHist(blurCPU, equalCPU);
        erode(equalCPU, morphCPU, kernel);
        dilate(morphCPU, morphCPU, kernel);
        Canny(morphCPU, edgesCPU, 50, 150);

        auto finCPU = chrono::high_resolution_clock::now();

        double tiempoCPU =
            chrono::duration<double, milli>(finCPU - inicioCPU).count();

        // ==========================
        // PIPELINE GPU-ONLY
        // Se sube una vez a GPU,
        // se procesa todo en GpuMat,
        // y se descarga solo al final.
        // ==========================
        cuda::GpuMat d_frame, d_gray, d_blur, d_equal, d_erode, d_dilate, d_edges;
        Mat edgesGPU;

        auto inicioGPU = chrono::high_resolution_clock::now();

        d_frame.upload(frameResize, stream);

        cuda::cvtColor(d_frame, d_gray, COLOR_BGR2GRAY, 0, stream);

        filtroGaussianoGPU->apply(d_gray, d_blur, stream);

        cuda::equalizeHist(d_blur, d_equal, stream);

        erosionGPU->apply(d_equal, d_erode, stream);
        dilatacionGPU->apply(d_erode, d_dilate, stream);

        cannyGPU->detect(d_dilate, d_edges, stream);

        d_edges.download(edgesGPU, stream);

        stream.waitForCompletion();

        auto finGPU = chrono::high_resolution_clock::now();

        double tiempoGPU =
            chrono::duration<double, milli>(finGPU - inicioGPU).count();

        // Evitar contar los primeros frames porque la GPU puede calentarse/inicializar contexto CUDA
        if (frameCount > warmup) {
            totalCpuMs += tiempoCPU;
            totalGpuMs += tiempoGPU;
            framesValidos++;

            csv << frameCount << ","
                << fixed << setprecision(4) << tiempoCPU << ","
                << fixed << setprecision(4) << tiempoGPU << "\n";
        }

        // ==========================
        // Visualización
        // ==========================
        Mat cpuColor, gpuColor, combinado;

        cvtColor(edgesCPU, cpuColor, COLOR_GRAY2BGR);
        cvtColor(edgesGPU, gpuColor, COLOR_GRAY2BGR);

        putText(cpuColor,
                "CPU: " + to_string(tiempoCPU).substr(0, 5) + " ms",
                Point(20, 35),
                FONT_HERSHEY_SIMPLEX,
                0.8,
                Scalar(0, 0, 255),
                2);

        putText(gpuColor,
                "GPU-only: " + to_string(tiempoGPU).substr(0, 5) + " ms",
                Point(20, 35),
                FONT_HERSHEY_SIMPLEX,
                0.8,
                Scalar(0, 255, 0),
                2);

        hconcat(cpuColor, gpuColor, combinado);

        imshow("Parte 1C - CPU vs GPU", combinado);

        if (frameCount == 30 || frameCount == 100) {
            imwrite("captura_parte_1c_frame_" + to_string(frameCount) + ".png", combinado);
        }

        char tecla = static_cast<char>(waitKey(1));

        if (tecla == 27 || tecla == 'q') {
            break;
        }
    }

    csv.close();

    double promedioCpu = totalCpuMs / framesValidos;
    double promedioGpu = totalGpuMs / framesValidos;

    double fpsCpu = 1000.0 / promedioCpu;
    double fpsGpu = 1000.0 / promedioGpu;

    double aceleracion = promedioCpu / promedioGpu;

    cout << "\nRESULTADOS PARTE 1C" << endl;
    cout << "Frames analizados: " << framesValidos << endl;
    cout << "Tiempo promedio CPU: " << promedioCpu << " ms/frame" << endl;
    cout << "Tiempo promedio GPU: " << promedioGpu << " ms/frame" << endl;
    cout << "FPS CPU: " << fpsCpu << endl;
    cout << "FPS GPU: " << fpsGpu << endl;
    cout << "Aceleracion GPU vs CPU: " << aceleracion << "x" << endl;

    ofstream resumen("resumen_parte_1c.txt");
    resumen << "===== RESULTADOS PARTE 1C =====\n";
    resumen << "Frames analizados: " << framesValidos << "\n";
    resumen << "Tiempo promedio CPU: " << promedioCpu << " ms/frame\n";
    resumen << "Tiempo promedio GPU: " << promedioGpu << " ms/frame\n";
    resumen << "FPS CPU: " << fpsCpu << "\n";
    resumen << "FPS GPU: " << fpsGpu << "\n";
    resumen << "Aceleracion GPU vs CPU: " << aceleracion << "x\n";
    resumen.close();

    cap.release();
    destroyAllWindows();

    return 0;
}