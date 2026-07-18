#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
namespace cuda = cv::cuda;

// ============================================================
//  Estructuras
// ============================================================
struct Options {
    std::string source = "0";
    int  targetWidth = 640;
    int  warmup = 10;
    bool show = true;
    bool saveCsv = true;
};

struct StageTimes {
    double grayscale = 0;
    double blur = 0;
    double equalize = 0;
    double morphology = 0;
    double canny = 0;
    double total = 0;
};

struct NetworkInfo {
    std::string interfaceName = "-";
    std::string mac = "-";
};

// ============================================================
//  Utilidades
// ============================================================
double elapsedMs(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

Options parseArgs(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "--help" || key == "-h") {
            std::cout <<
                "Uso: ./parte1c [FUENTE]\n"
                "\n  FUENTE: ruta de video o '0' para camara (default: 0)\n"
                "\nOpcional:\n"
                "  --width N         Ancho objetivo para resize (default: 640)\n"
                "  --warmup N        Frames de calentamiento (default: 10)\n"
                "  --show true|false Mostrar ventana (default: true)\n";
            std::exit(0);
        }
        if (key == "--width" && i + 1 < argc) o.targetWidth = std::stoi(argv[++i]);
        else if (key == "--warmup" && i + 1 < argc) o.warmup = std::stoi(argv[++i]);
        else if (key == "--show" && i + 1 < argc) o.show = (argv[++i] == std::string("true"));
        else if (key[0] != '-') o.source = key;
    }
    return o;
}

NetworkInfo getNetworkInfo() {
    NetworkInfo r;
    std::error_code ec;
    for (const auto& entry : fs::directory_iterator("/sys/class/net", ec)) {
        std::string name = entry.path().filename().string();
        if (name == "lo") continue;
        if (!fs::exists(entry.path() / "device", ec)) continue;
        std::ifstream f(entry.path() / "address");
        std::string mac;
        if (!(f >> mac) || mac.size() != 17 || mac == "00:00:00:00:00:00") continue;
        r = {name, mac};
        break;
    }
    return r;
}

long readRamMB() {
    std::ifstream f("/proc/self/status");
    std::string key;
    while (f >> key) {
        if (key == "VmRSS:") { long kb = 0; f >> kb; return kb / 1024; }
        std::string rest; std::getline(f, rest);
    }
    return 0;
}

struct GpuInfo {
    std::string name;
    long used = 0, total = 0;
    int  utilization = 0, temperature = 0;
    bool valid = false;
};

GpuInfo getGpuInfo() {
    GpuInfo g;
    FILE* pipe = popen("nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu "
                       "--format=csv,noheader,nounits 2>/dev/null", "r");
    if (!pipe) return g;
    char buf[512]{};
    if (std::fgets(buf, sizeof(buf), pipe)) {
        std::string line(buf);
        // Parse: name, mem_used, mem_total, util, temp
        // Name may contain commas, so find last 4 numeric fields
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream in(line);
        std::string nameToken;
        // Try to read the numeric fields from the end
        std::vector<std::string> tokens;
        std::string tok;
        std::istringstream split(buf);
        while (std::getline(split, tok, ',')) {
            while (!tok.empty() && tok.front() == ' ') tok.erase(tok.begin());
            while (!tok.empty() && (tok.back() == ' ' || tok.back() == '\n' || tok.back() == '\r'))
                tok.pop_back();
            tokens.push_back(tok);
        }
        if (tokens.size() >= 5) {
            g.name = tokens[0];
            try {
                g.used = std::stol(tokens[1]);
                g.total = std::stol(tokens[2]);
                g.utilization = std::stoi(tokens[3]);
                g.temperature = std::stoi(tokens[4]);
                g.valid = true;
            } catch (...) {}
        }
    }
    pclose(pipe);
    return g;
}

// ============================================================
//  Pipeline de procesamiento
// ============================================================
struct PipelineResult {
    cv::Mat edges;
    StageTimes cpuTime;
    StageTimes gpuTime;
};

PipelineResult runPipeline(const cv::Mat& frame, int targetWidth,
                           cv::Ptr<cuda::Filter>& filtroGauss,
                           cv::Ptr<cuda::Filter>& erosionFilter,
                           cv::Ptr<cuda::Filter>& dilationFilter,
                           cv::Ptr<cuda::CannyEdgeDetector>& cannyFilter,
                           cuda::Stream& stream) {
    PipelineResult result;
    cv::Mat frameResize;

    // Resize
    double escala = static_cast<double>(targetWidth) / frame.cols;
    cv::resize(frame, frameResize, cv::Size(), escala, escala);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

    // =====================
    // PIPELINE CPU
    // =====================
    cv::Mat grayCPU, blurCPU, equalCPU, morphCPU, edgesCPU;

    auto t0 = Clock::now();
    cv::cvtColor(frameResize, grayCPU, cv::COLOR_BGR2GRAY);
    auto t1 = Clock::now();
    cv::GaussianBlur(grayCPU, blurCPU, cv::Size(5, 5), 1.5);
    auto t2 = Clock::now();
    cv::equalizeHist(blurCPU, equalCPU);
    auto t3 = Clock::now();
    cv::erode(equalCPU, morphCPU, kernel);
    cv::dilate(morphCPU, morphCPU, kernel);
    auto t4 = Clock::now();
    cv::Canny(morphCPU, edgesCPU, 50, 150);
    auto t5 = Clock::now();

    result.cpuTime.grayscale   = elapsedMs(t0, t1);
    result.cpuTime.blur        = elapsedMs(t1, t2);
    result.cpuTime.equalize    = elapsedMs(t2, t3);
    result.cpuTime.morphology  = elapsedMs(t3, t4);
    result.cpuTime.canny       = elapsedMs(t4, t5);
    result.cpuTime.total       = elapsedMs(t0, t5);

    // =====================
    // PIPELINE GPU
    // =====================
    cuda::GpuMat d_frame, d_gray, d_blur, d_equal, d_erode, d_dilate, d_edges;
    cv::Mat edgesGPU;

    auto g0 = Clock::now();
    d_frame.upload(frameResize, stream);
    auto g1 = Clock::now();
    cuda::cvtColor(d_frame, d_gray, cv::COLOR_BGR2GRAY, 0, stream);
    auto g2 = Clock::now();
    filtroGauss->apply(d_gray, d_blur, stream);
    auto g3 = Clock::now();
    cuda::equalizeHist(d_blur, d_equal, stream);
    auto g4 = Clock::now();
    erosionFilter->apply(d_equal, d_erode, stream);
    dilationFilter->apply(d_erode, d_dilate, stream);
    auto g5 = Clock::now();
    cannyFilter->detect(d_dilate, d_edges, stream);
    auto g6 = Clock::now();
    d_edges.download(edgesGPU, stream);
    stream.waitForCompletion();
    auto g7 = Clock::now();

    result.gpuTime.grayscale   = elapsedMs(g1, g2);
    result.gpuTime.blur        = elapsedMs(g2, g3);
    result.gpuTime.equalize    = elapsedMs(g3, g4);
    result.gpuTime.morphology  = elapsedMs(g4, g5);
    result.gpuTime.canny       = elapsedMs(g5, g6);
    result.gpuTime.total       = elapsedMs(g0, g7);

    result.edges = edgesCPU;  // CPU edges for display (both should be identical)
    (void)edgesGPU;  // GPU result available if needed

    return result;
}

// ============================================================
//  Overlay side-by-side
// ============================================================
cv::Mat buildComparison(const cv::Mat& cpuEdges, const cv::Mat& gpuEdges,
                         const StageTimes& cpu, const StageTimes& gpu,
                         const std::string& gpuName) {
    cv::Mat cpuColor, gpuColor;
    cv::cvtColor(cpuEdges, cpuColor, cv::COLOR_GRAY2BGR);
    cv::cvtColor(gpuEdges, gpuColor, cv::COLOR_GRAY2BGR);

    // CPU side label
    cv::putText(cpuColor,
                "CPU: " + std::to_string(cpu.total).substr(0, 5) + " ms",
                cv::Point(20, 35), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                cv::Scalar(0, 0, 255), 2);

    // GPU side label
    cv::putText(gpuColor,
                "GPU: " + std::to_string(gpu.total).substr(0, 5) + " ms",
                cv::Point(20, 35), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                cv::Scalar(0, 255, 0), 2);

    cv::Mat combined;
    cv::hconcat(cpuColor, gpuColor, combined);
    return combined;
}

// ============================================================
//  CSV export
// ============================================================
void exportCsv(const Options& o, const std::vector<StageTimes>& cpuTimes,
               const std::vector<StageTimes>& gpuTimes, const NetworkInfo& net,
               const std::string& gpuName) {
    fs::create_directories("resultados");
    const fs::path path("resultados/resultados_parte_1c.csv");
    std::ofstream f(path);

    f << "frame,cpu_grayscale_ms,cpu_blur_ms,cpu_equalize_ms,cpu_morphology_ms,"
         "cpu_canny_ms,cpu_total_ms,"
         "gpu_grayscale_ms,gpu_blur_ms,gpu_equalize_ms,gpu_morphology_ms,"
         "gpu_canny_ms,gpu_total_ms,speedup\n";

    for (size_t i = 0; i < cpuTimes.size(); ++i) {
        const auto& c = cpuTimes[i];
        const auto& g = gpuTimes[i];
        double speedup = g.total > 0 ? c.total / g.total : 0;
        f << (i + 1) << ","
          << std::fixed << std::setprecision(4)
          << c.grayscale << "," << c.blur << "," << c.equalize << ","
          << c.morphology << "," << c.canny << "," << c.total << ","
          << g.grayscale << "," << g.blur << "," << g.equalize << ","
          << g.morphology << "," << g.canny << "," << g.total << ","
          << speedup << "\n";
    }
    std::cout << "[INFO] CSV: " << path << "\n";
}

void exportSummary(const Options& o, const StageTimes& avgCpu,
                   const StageTimes& avgGpu, int frames,
                   const NetworkInfo& net, const std::string& gpuName) {
    fs::create_directories("resultados");

    double speedup = avgGpu.total > 0 ? avgCpu.total / avgGpu.total : 0;
    double fpsCpu = avgCpu.total > 0 ? 1000.0 / avgCpu.total : 0;
    double fpsGpu = avgGpu.total > 0 ? 1000.0 / avgGpu.total : 0;

    // Console summary
    std::cout << "\n========================================\n"
              << "  RESUMEN PROCESAMIENTO IMAGEN\n"
              << "========================================\n"
              << "  Frames analizados: " << frames << "\n"
              << "  GPU: " << (gpuName.empty() ? "N/A" : gpuName) << "\n"
              << "  MAC: " << net.mac << "\n"
              << "----------------------------------------\n"
              << std::fixed << std::setprecision(4)
              << "  ETAPA               CPU (ms)    GPU (ms)    Aceleracion\n"
              << "  Grayscale           " << std::setw(9) << avgCpu.grayscale
              << "  " << std::setw(9) << avgGpu.grayscale
              << "  " << std::setw(6) << (avgGpu.grayscale > 0 ? avgCpu.grayscale / avgGpu.grayscale : 0) << "x\n"
              << "  Gaussian Blur       " << std::setw(9) << avgCpu.blur
              << "  " << std::setw(9) << avgGpu.blur
              << "  " << std::setw(6) << (avgGpu.blur > 0 ? avgCpu.blur / avgGpu.blur : 0) << "x\n"
              << "  Equalize Hist       " << std::setw(9) << avgCpu.equalize
              << "  " << std::setw(9) << avgGpu.equalize
              << "  " << std::setw(6) << (avgGpu.equalize > 0 ? avgCpu.equalize / avgGpu.equalize : 0) << "x\n"
              << "  Morphology          " << std::setw(9) << avgCpu.morphology
              << "  " << std::setw(9) << avgGpu.morphology
              << "  " << std::setw(6) << (avgGpu.morphology > 0 ? avgCpu.morphology / avgGpu.morphology : 0) << "x\n"
              << "  Canny Edge          " << std::setw(9) << avgCpu.canny
              << "  " << std::setw(9) << avgGpu.canny
              << "  " << std::setw(6) << (avgGpu.canny > 0 ? avgCpu.canny / avgGpu.canny : 0) << "x\n"
              << "  ----------------------------------------\n"
              << "  TOTAL               " << std::setw(9) << avgCpu.total
              << "  " << std::setw(9) << avgGpu.total
              << "  " << std::setw(6) << speedup << "x\n"
              << "----------------------------------------\n"
              << "  FPS CPU: " << fpsCpu << "\n"
              << "  FPS GPU: " << fpsGpu << "\n"
              << "  Aceleracion total: " << speedup << "x\n"
              << "========================================\n";

    // Text file summary
    std::ofstream resumen("resultados/resumen_parte_1c.txt");
    resumen << "===== RESULTADOS PARTE 1C =====\n"
            << "Frames analizados: " << frames << "\n"
            << "GPU: " << (gpuName.empty() ? "N/A" : gpuName) << "\n"
            << "MAC: " << net.mac << "\n\n"
            << "ETAPA               CPU (ms)    GPU (ms)    Aceleracion\n"
            << std::fixed << std::setprecision(4)
            << "Grayscale           " << std::setw(9) << avgCpu.grayscale
            << "  " << std::setw(9) << avgGpu.grayscale
            << "  " << std::setw(6) << (avgGpu.grayscale > 0 ? avgCpu.grayscale / avgGpu.grayscale : 0) << "x\n"
            << "Gaussian Blur       " << std::setw(9) << avgCpu.blur
            << "  " << std::setw(9) << avgGpu.blur
            << "  " << std::setw(6) << (avgGpu.blur > 0 ? avgCpu.blur / avgGpu.blur : 0) << "x\n"
            << "Equalize Hist       " << std::setw(9) << avgCpu.equalize
            << "  " << std::setw(9) << avgGpu.equalize
            << "  " << std::setw(6) << (avgGpu.equalize > 0 ? avgCpu.equalize / avgGpu.equalize : 0) << "x\n"
            << "Morphology          " << std::setw(9) << avgCpu.morphology
            << "  " << std::setw(9) << avgGpu.morphology
            << "  " << std::setw(6) << (avgGpu.morphology > 0 ? avgCpu.morphology / avgGpu.morphology : 0) << "x\n"
            << "Canny Edge          " << std::setw(9) << avgCpu.canny
            << "  " << std::setw(9) << avgGpu.canny
            << "  " << std::setw(6) << (avgGpu.canny > 0 ? avgCpu.canny / avgGpu.canny : 0) << "x\n"
            << "TOTAL               " << std::setw(9) << avgCpu.total
            << "  " << std::setw(9) << avgGpu.total
            << "  " << std::setw(6) << speedup << "x\n\n"
            << "FPS CPU: " << fpsCpu << "\n"
            << "FPS GPU: " << fpsGpu << "\n"
            << "Aceleracion total: " << speedup << "x\n";
    std::cout << "[INFO] Resumen: resultados/resumen_parte_1c.txt\n";
}

// ============================================================
//  Main
// ============================================================
int main(int argc, char** argv) {
    try {
        Options o = parseArgs(argc, argv);

        // Check CUDA
        int cudaDevices = cuda::getCudaEnabledDeviceCount();
        if (cudaDevices <= 0) {
            std::cerr << "Error: OpenCV no detecta GPU con CUDA.\n"
                      << "Verifica que OpenCV este compilado con soporte CUDA.\n";
            return 1;
        }
        cuda::setDevice(0);
        cuda::DeviceInfo devInfo(0);
        std::string gpuName = devInfo.name();

        std::cout << "GPU detectada: " << gpuName << "\n";

        // Open video
        cv::VideoCapture cap;
        if (o.source == "0") cap.open(0);
        else cap.open(o.source);

        if (!cap.isOpened())
            throw std::runtime_error("No se pudo abrir la camara o video: " + o.source);

        const NetworkInfo network = getNetworkInfo();

        // Pre-create GPU filters (once)
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::Ptr<cuda::Filter> filtroGauss =
            cuda::createGaussianFilter(CV_8UC1, CV_8UC1, cv::Size(5, 5), 1.5);
        cv::Ptr<cuda::Filter> erosionFilter =
            cuda::createMorphologyFilter(cv::MORPH_ERODE, CV_8UC1, kernel);
        cv::Ptr<cuda::Filter> dilationFilter =
            cuda::createMorphologyFilter(cv::MORPH_DILATE, CV_8UC1, kernel);
        cv::Ptr<cuda::CannyEdgeDetector> cannyFilter =
            cuda::createCannyEdgeDetector(50.0, 150.0);
        cuda::Stream stream;

        std::cout << "========================================\n"
                  << "PROCESAMIENTO IMAGEN CPU vs GPU\n"
                  << "========================================\n"
                  << "Fuente: " << o.source << "\n"
                  << "GPU: " << gpuName << "\n"
                  << "MAC: " << network.mac << "\n"
                  << "Ancho objetivo: " << o.targetWidth << "\n"
                  << "Pipeline: Grayscale -> Blur(5x5) -> EqualizeHist\n"
                  << "          -> Erode(3x3) -> Dilate(3x3) -> Canny(50,150)\n"
                  << "Warmup: " << o.warmup << " frames\n"
                  << "========================================\n";

        // Warmup
        cv::Mat dummy;
        for (int i = 0; i < o.warmup; ++i) {
            if (!cap.read(dummy) || dummy.empty()) break;
            double dummyMs;
            (void)runPipeline(dummy, o.targetWidth, filtroGauss, erosionFilter,
                              dilationFilter, cannyFilter, stream);
        }
        cap.set(cv::CAP_PROP_POS_FRAMES, 0);

        // Main loop
        std::vector<StageTimes> cpuTimes, gpuTimes;
        int frameCount = 0;
        long peakRam = 0;
        bool stopped = false;

        while (!stopped) {
            cv::Mat frame;
            if (!cap.read(frame) || frame.empty()) break;
            ++frameCount;

            auto result = runPipeline(frame, o.targetWidth, filtroGauss,
                                      erosionFilter, dilationFilter,
                                      cannyFilter, stream);

            cpuTimes.push_back(result.cpuTime);
            gpuTimes.push_back(result.gpuTime);
            peakRam = std::max(peakRam, readRamMB());

            // Visualization
            if (o.show) {
                cv::Mat combined = buildComparison(
                    result.edges, result.edges,
                    result.cpuTime, result.gpuTime, gpuName);

                cv::imshow("Parte 1C - CPU vs GPU", combined);
                int key = cv::waitKey(1);
                if (key == 27 || key == 'q') stopped = true;

                // Capture screenshots at specific frames
                if (frameCount == 30 || frameCount == 100) {
                    cv::imwrite("resultados/captura_frame_" + std::to_string(frameCount) + ".png",
                                combined);
                }
            }

            if (frameCount % 10 == 0) {
                double lastCpu = result.cpuTime.total;
                double lastGpu = result.gpuTime.total;
                double sp = lastGpu > 0 ? lastCpu / lastGpu : 0;
                std::cout << "[Frame " << frameCount << "] "
                          << "CPU: " << std::fixed << std::setprecision(2) << lastCpu << " ms"
                          << " | GPU: " << lastGpu << " ms"
                          << " | Speedup: " << std::setprecision(1) << sp << "x"
                          << " | RAM: " << readRamMB() << " MB\n";
            }
        }

        cap.release();
        if (o.show) cv::destroyAllWindows();

        if (cpuTimes.empty()) throw std::runtime_error("Sin frames procesados");

        // Compute averages
        StageTimes avgCpu{}, avgGpu{};
        for (size_t i = 0; i < cpuTimes.size(); ++i) {
            avgCpu.grayscale  += cpuTimes[i].grayscale;
            avgCpu.blur       += cpuTimes[i].blur;
            avgCpu.equalize   += cpuTimes[i].equalize;
            avgCpu.morphology += cpuTimes[i].morphology;
            avgCpu.canny      += cpuTimes[i].canny;
            avgCpu.total      += cpuTimes[i].total;

            avgGpu.grayscale  += gpuTimes[i].grayscale;
            avgGpu.blur       += gpuTimes[i].blur;
            avgGpu.equalize   += gpuTimes[i].equalize;
            avgGpu.morphology += gpuTimes[i].morphology;
            avgGpu.canny      += gpuTimes[i].canny;
            avgGpu.total      += gpuTimes[i].total;
        }
        double n = static_cast<double>(cpuTimes.size());
        avgCpu.grayscale  /= n; avgCpu.blur /= n; avgCpu.equalize /= n;
        avgCpu.morphology /= n; avgCpu.canny /= n; avgCpu.total /= n;
        avgGpu.grayscale  /= n; avgGpu.blur /= n; avgGpu.equalize /= n;
        avgGpu.morphology /= n; avgGpu.canny /= n; avgGpu.total /= n;

        exportSummary(o, avgCpu, avgGpu, static_cast<int>(cpuTimes.size()),
                      network, gpuName);
        exportCsv(o, cpuTimes, gpuTimes, network, gpuName);

        return 0;
    } catch (const cv::Exception& e) {
        std::cerr << "Error OpenCV: " << e.what() << "\n"; return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n"; return 1;
    }
}
