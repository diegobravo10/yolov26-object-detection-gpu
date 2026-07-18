#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <chrono>
#include <numeric>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

// ============================================================
//  Estructuras
// ============================================================
struct Options {
    std::string model  = "modelo/realesr-general-x4v3.onnx";
    std::string input;
    std::string output = "results/output.mp4";
    std::string device = "cpu";
    int  tile     = 128;
    int  tilePad  = 10;
    int  maxFrames = 0;
    bool show     = true;
    bool save     = true;
};

struct GpuStats {
    long used = 0, total = 0;
    int  utilization = 0, temperature = 0;
    bool valid = false;
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
                "Uso: ./realesrgan_bench [OPCIONES]\n"
                "\nRequerido:\n"
                "  --input RUTA          Video de entrada\n"
                "\nOpcional:\n"
                "  --model RUTA          Modelo ONNX (default: " << o.model << ")\n"
                "  --output RUTA         Video de salida\n"
                "  --device cpu|cuda|cuda_fp16   (default: cpu)\n"
                "  --tile N              Tamano de tile (default: 128)\n"
                "  --tile-pad N          Padding de tile (default: 10)\n"
                "  --max-frames N        Max frames (0=todos)\n"
                "  --show true|false     Mostrar ventana\n"
                "  --save true|false     Guardar video\n";
            std::exit(0);
        }
        if (i + 1 >= argc) throw std::runtime_error("Falta valor para " + key);
        std::string val = argv[++i];
        if      (key == "--model")      o.model = val;
        else if (key == "--input")      o.input = val;
        else if (key == "--output")     o.output = val;
        else if (key == "--device")     o.device = val;
        else if (key == "--tile")       o.tile = std::stoi(val);
        else if (key == "--tile-pad")   o.tilePad = std::stoi(val);
        else if (key == "--max-frames") o.maxFrames = std::stoi(val);
        else if (key == "--show")       o.show = (val == "true" || val == "1");
        else if (key == "--save")       o.save = (val == "true" || val == "1");
        else throw std::runtime_error("Argumento desconocido: " + key);
    }
    if (o.input.empty()) throw std::runtime_error("--input es obligatorio");
    if (o.device != "cpu" && o.device != "cuda" && o.device != "cuda_fp16")
        throw std::runtime_error("--device debe ser cpu, cuda o cuda_fp16");
    return o;
}

std::string deviceLabel(const std::string& d) {
    if (d == "cuda") return "GPU CUDA FP32";
    if (d == "cuda_fp16") return "GPU CUDA FP16";
    return "CPU";
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

GpuStats readGpuStats() {
    GpuStats s;
    FILE* pipe = popen("nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu "
                       "--format=csv,noheader,nounits 2>/dev/null", "r");
    if (!pipe) return s;
    char buf[256]{};
    if (std::fgets(buf, sizeof(buf), pipe)) {
        std::string line(buf);
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream in(line);
        s.valid = static_cast<bool>(in >> s.used >> s.total >> s.utilization >> s.temperature);
    }
    pclose(pipe);
    return s;
}

// ============================================================
//  Tensor to BGR (Real-ESRGAN output: [1,3,H,W] float32 RGB)
// ============================================================
cv::Mat tensorToBgr(const cv::Mat& output, int expectedW, int expectedH) {
    if (output.dims != 4 || output.size[0] != 1 || output.size[1] != 3)
        throw std::runtime_error("Salida ONNX inesperada para Real-ESRGAN");
    int h = output.size[2], w = output.size[3];
    if (h != expectedH || w != expectedW)
        throw std::runtime_error("Resolucion de salida incorrecta");

    cv::Mat cont = output.isContinuous() ? output : output.clone();
    std::vector<cv::Mat> channels;
    for (int c = 0; c < 3; ++c)
        channels.emplace_back(h, w, CV_32F, cont.ptr<float>(0, c));

    cv::Mat rgbFloat, rgb8, bgr;
    cv::merge(channels, rgbFloat);
    cv::max(rgbFloat, 0.0, rgbFloat);
    cv::min(rgbFloat, 1.0, rgbFloat);
    rgbFloat.convertTo(rgb8, CV_8UC3, 255.0);
    cv::cvtColor(rgb8, bgr, cv::COLOR_RGB2BGR);
    return bgr;
}

// ============================================================
//  Tile processing
// ============================================================
cv::Mat buildPaddedTile(const cv::Mat& frame, int x, int y,
                         int coreW, int coreH, int tile, int pad) {
    int x0 = std::max(0, x - pad), y0 = std::max(0, y - pad);
    int x1 = std::min(frame.cols, x + tile + pad);
    int y1 = std::min(frame.rows, y + tile + pad);
    cv::Mat crop = frame(cv::Rect(x0, y0, x1 - x0, y1 - y0));
    cv::Mat padded;
    cv::copyMakeBorder(crop, padded,
                       y0 - (y - pad), (y + tile + pad) - y1,
                       x0 - (x - pad), (x + tile + pad) - x1,
                       cv::BORDER_REFLECT_101);
    return padded;
}

cv::Mat upscaleFrame(cv::dnn::Net& net, const cv::Mat& frame,
                      int tile, int pad, double& inferenceMs) {
    constexpr int scale = 4;
    cv::Mat result(frame.rows * scale, frame.cols * scale, CV_8UC3);
    inferenceMs = 0.0;

    for (int y = 0; y < frame.rows; y += tile) {
        int coreH = std::min(tile, frame.rows - y);
        for (int x = 0; x < frame.cols; x += tile) {
            int coreW = std::min(tile, frame.cols - x);
            cv::Mat padded = buildPaddedTile(frame, x, y, coreW, coreH, tile, pad);
            cv::Mat blob = cv::dnn::blobFromImage(padded, 1.0/255.0, cv::Size(),
                                                   cv::Scalar(), true, false, CV_32F);
            net.setInput(blob);
            auto t0 = Clock::now();
            cv::Mat output = net.forward();
            inferenceMs += elapsedMs(t0, Clock::now());

            cv::Mat tileBgr = tensorToBgr(output, padded.cols * scale, padded.rows * scale);
            cv::Rect valid(pad * scale, pad * scale, coreW * scale, coreH * scale);
            cv::Rect dest(x * scale, y * scale, coreW * scale, coreH * scale);
            tileBgr(valid).copyTo(result(dest));
        }
    }
    return result;
}

// ============================================================
//  Overlay
// ============================================================
void drawOverlay(cv::Mat& image, const std::string& device,
                  double fps, long ramMb,
                  int inputW, int inputH, const NetworkInfo& net) {
    std::ostringstream ssFps, ssRam;
    ssFps << std::fixed << std::setprecision(2) << "FPS: " << fps;
    ssRam << "RAM: " << ramMb << " MB";

    std::vector<std::pair<std::string, cv::Scalar>> lines = {
        {"Real-ESRGAN x4 | " + device, {100, 220, 255}},
        {ssFps.str(), {255, 220, 80}},
        {ssRam.str(), {200, 200, 200}},
        {"Entrada: " + std::to_string(inputW) + "x" + std::to_string(inputH) +
         " -> Salida: " + std::to_string(image.cols) + "x" + std::to_string(image.rows),
         {100, 255, 150}},
        {"MAC: " + net.mac, {180, 180, 255}},
    };

    double scale = std::clamp(image.cols / 1600.0, 0.55, 1.1);
    int thick = std::max(1, static_cast<int>(scale * 2));
    int dy = static_cast<int>(30 * scale) + 8;
    int margin = static_cast<int>(10 * scale);
    int padPx = static_cast<int>(6 * scale);

    int maxW = 0;
    for (auto& [txt, _] : lines) {
        int bl = 0;
        auto sz = cv::getTextSize(txt, cv::FONT_HERSHEY_SIMPLEX, scale, thick, &bl);
        maxW = std::max(maxW, sz.width);
    }

    int boxH = static_cast<int>(lines.size()) * dy + padPx * 2;
    cv::Rect roiRect(margin, margin,
                     std::min(maxW + padPx * 2, image.cols - margin * 2),
                     std::min(boxH, image.rows - margin * 2));
    if (roiRect.width > 0 && roiRect.height > 0) {
        cv::Mat roi = image(roiRect);
        cv::Mat dark(roi.size(), roi.type(), cv::Scalar(0, 0, 0));
        cv::addWeighted(roi, 0.35, dark, 0.65, 0, roi);
    }

    int y = margin + padPx + dy - static_cast<int>(4 * scale);
    for (auto& [txt, col] : lines) {
        cv::putText(image, txt, cv::Point(margin + padPx, y),
                    cv::FONT_HERSHEY_SIMPLEX, scale, col, thick, cv::LINE_AA);
        y += dy;
    }
}

// ============================================================
//  CSV export
// ============================================================
void exportCsv(const Options& o, int inputW, int inputH, int frames,
               double avgFps, double avgInferMs, double avgTotalMs,
               long peakRam, long peakVram, const NetworkInfo& net) {
    fs::create_directories("resultados");
    const fs::path path("resultados/superres_benchmark.csv");
    bool header = !fs::exists(path) || fs::file_size(path) == 0;
    std::ofstream f(path, std::ios::app);

    if (header)
        f << "timestamp,model,device,tile,tile_pad,input_width,input_height,"
             "output_width,output_height,frames,average_fps,average_inference_ms,"
             "average_total_ms,peak_ram_mb,peak_vram_mb,mac_address\n";

    const std::time_t now = std::time(nullptr);
    std::tm tm{};
    localtime_r(&now, &tm);

    f << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S") << ','
      << fs::path(o.model).stem().string() << ',' << deviceLabel(o.device) << ','
      << o.tile << ',' << o.tilePad << ','
      << inputW << ',' << inputH << ',' << inputW * 4 << ',' << inputH * 4 << ','
      << frames << ',' << std::fixed << std::setprecision(2)
      << avgFps << ',' << avgInferMs << ',' << avgTotalMs << ','
      << peakRam << ',' << peakVram << ',' << net.mac << '\n';
    std::cout << "[INFO] CSV: " << path << "\n";
}

// ============================================================
//  Main
// ============================================================
int main(int argc, char** argv) {
    try {
        const Options o = parseArgs(argc, argv);
        if (!fs::is_regular_file(o.model))
            throw std::runtime_error("Modelo no encontrado: " + o.model);
        if (!fs::is_regular_file(o.input))
            throw std::runtime_error("Video no encontrado: " + o.input);

        fs::create_directories("resultados");
        const NetworkInfo network = getNetworkInfo();
        const bool useCuda = o.device != "cpu";
        std::string gpuName;

        if (useCuda) {
            int count = cv::cuda::getCudaEnabledDeviceCount();
            if (count <= 0)
                throw std::runtime_error("CUDA no disponible en OpenCV");
            cv::cuda::DeviceInfo info(0);
            gpuName = info.name();
        }

        cv::dnn::Net net;
        try { net = cv::dnn::readNetFromONNX(o.model); }
        catch (const cv::Exception& e) {
            throw std::runtime_error("Error ONNX: " + std::string(e.what()));
        }
        if (net.empty()) throw std::runtime_error("ONNX vacio");

        if (useCuda) {
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            net.setPreferableTarget(o.device == "cuda_fp16"
                                    ? cv::dnn::DNN_TARGET_CUDA_FP16
                                    : cv::dnn::DNN_TARGET_CUDA);
        } else {
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        }

        cv::VideoCapture cap(o.input);
        if (!cap.isOpened()) throw std::runtime_error("No se pudo abrir: " + o.input);
        int inputW = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int inputH = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        double srcFps = cap.get(cv::CAP_PROP_FPS);
        if (!(srcFps > 0)) srcFps = 30.0;
        int reportedFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

        cv::VideoWriter writer;
        if (o.save) {
            fs::path parent = fs::path(o.output).parent_path();
            if (!parent.empty()) fs::create_directories(parent);
            writer.open(o.output, cv::VideoWriter::fourcc('m','p','4','v'),
                        srcFps, cv::Size(inputW * 4, inputH * 4));
            if (!writer.isOpened())
                writer.open(o.output, cv::VideoWriter::fourcc('a','v','c','1'),
                            srcFps, cv::Size(inputW * 4, inputH * 4));
        }

        std::cout << "========================================\n"
                  << "REAL-ESRGAN VIDEO SUPER RESOLUTION\n"
                  << "========================================\n"
                  << "Modelo: " << o.model << "\n"
                  << "Video: " << o.input << " [" << inputW << "x" << inputH
                  << " @ " << std::fixed << std::setprecision(1) << srcFps << " FPS]\n"
                  << "Frames: " << reportedFrames << "\n"
                  << "Dispositivo: " << deviceLabel(o.device) << "\n"
                  << "GPU: " << (gpuName.empty() ? "N/A" : gpuName) << "\n"
                  << "MAC: " << network.mac << "\n"
                  << "Tile: " << o.tile << " | Padding: " << o.tilePad << "\n"
                  << "Escala: x4 (" << inputW << "x" << inputH
                  << " -> " << inputW*4 << "x" << inputH*4 << ")\n"
                  << "OpenCV: " << CV_VERSION << "\n"
                  << "========================================\n";

        std::vector<double> fpsValues;
        double sumInfer = 0.0, sumTotal = 0.0;
        long peakRam = 0, peakVram = 0;
        int processed = 0;
        GpuStats gpu;
        bool stopped = false;

        while ((o.maxFrames == 0 || processed < o.maxFrames) && !stopped) {
            cv::Mat frame;
            if (!cap.read(frame) || frame.empty()) break;

            auto t0 = Clock::now();
            double inferMs = 0.0;
            cv::Mat enhanced = upscaleFrame(net, frame, o.tile, o.tilePad, inferMs);
            auto t1 = Clock::now();

            double totalMs = elapsedMs(t0, t1);
            double currentFps = totalMs > 0 ? 1000.0 / totalMs : 0;
            fpsValues.push_back(currentFps);
            sumInfer += inferMs;
            sumTotal += totalMs;
            ++processed;

            peakRam = std::max(peakRam, readRamMB());
            if (useCuda && (processed == 1 || processed % 10 == 0)) {
                gpu = readGpuStats();
                if (gpu.valid) peakVram = std::max(peakVram, gpu.used);
            }

            drawOverlay(enhanced, deviceLabel(o.device), currentFps,
                       readRamMB(), inputW, inputH, network);

            if (o.save && writer.isOpened()) writer.write(enhanced);

            if (o.show) {
                int targetH = std::min(enhanced.rows, 720);
                double ratioEnhanced = static_cast<double>(targetH) / enhanced.rows;
                cv::Mat enhancedResized;
                cv::resize(enhanced, enhancedResized, cv::Size(), ratioEnhanced, ratioEnhanced, cv::INTER_AREA);

                double ratioOrig = static_cast<double>(targetH) / frame.rows;
                cv::Mat origResized;
                cv::resize(frame, origResized, cv::Size(), ratioOrig, ratioOrig, cv::INTER_AREA);

                int gap = 4;
                cv::Mat combined(origResized.rows, origResized.cols + gap + enhancedResized.cols,
                                 origResized.type(), cv::Scalar(0, 0, 0));
                origResized.copyTo(combined(cv::Rect(0, 0, origResized.cols, origResized.rows)));
                enhancedResized.copyTo(combined(cv::Rect(origResized.cols + gap, 0,
                                                         enhancedResized.cols, enhancedResized.rows)));

                double ts = std::clamp(combined.cols / 1200.0, 0.5, 1.0);
                int thick = std::max(1, static_cast<int>(ts * 2));
                int bl = 0;
                cv::Size sz;
                int bottomY = combined.rows - 14;

                sz = cv::getTextSize("Original", cv::FONT_HERSHEY_SIMPLEX, ts, thick, &bl);
                cv::rectangle(combined,
                              cv::Point(8, bottomY - sz.height - 8),
                              cv::Point(8 + sz.width + 12, bottomY + 6),
                              cv::Scalar(0, 0, 0), cv::FILLED);
                cv::putText(combined, "Original", cv::Point(14, bottomY),
                            cv::FONT_HERSHEY_SIMPLEX, ts, {80, 200, 255}, thick, cv::LINE_AA);

                sz = cv::getTextSize("Mejorado", cv::FONT_HERSHEY_SIMPLEX, ts, thick, &bl);
                int mx = origResized.cols + gap + 8;
                cv::rectangle(combined,
                              cv::Point(mx, bottomY - sz.height - 8),
                              cv::Point(mx + sz.width + 12, bottomY + 6),
                              cv::Scalar(0, 0, 0), cv::FILLED);
                cv::putText(combined, "Mejorado", cv::Point(mx + 6, bottomY),
                            cv::FONT_HERSHEY_SIMPLEX, ts, {80, 255, 150}, thick, cv::LINE_AA);

                cv::imshow("Real-ESRGAN x4 | " + deviceLabel(o.device), combined);
                int key = cv::waitKey(1) & 0xff;
                if (key == 27 || key == 'q') stopped = true;
            }

            if (processed % 5 == 0) {
                std::cout << "[" << deviceLabel(o.device) << "] Frame " << processed
                          << " | FPS: " << std::fixed << std::setprecision(2) << currentFps
                          << " | Infer: " << inferMs << " ms"
                          << " | Total: " << totalMs << " ms"
                          << " | RAM: " << readRamMB() << " MB";
                if (useCuda && gpu.valid)
                    std::cout << " | VRAM: " << gpu.used << "/" << gpu.total << " MB";
                std::cout << "\n";
            }
        }

        if (o.show) cv::destroyAllWindows();
        writer.release();
        cap.release();

        if (processed == 0) throw std::runtime_error("Sin frames procesados");

        double avgFps = std::accumulate(fpsValues.begin(), fpsValues.end(), 0.0)
                        / fpsValues.size();
        auto [minIt, maxIt] = std::minmax_element(fpsValues.begin(), fpsValues.end());

        std::cout << "\n========================================\n"
                  << "  RESUMEN SUPER-RESOLUCION\n"
                  << "========================================\n"
                  << "  Dispositivo: " << deviceLabel(o.device) << "\n"
                  << "  Frames: " << processed << "\n"
                  << "  FPS promedio: " << std::fixed << std::setprecision(2) << avgFps << "\n"
                  << "  FPS minimo: " << *minIt << "\n"
                  << "  FPS maximo: " << *maxIt << "\n"
                  << "  Inferencia promedio: " << sumInfer / processed << " ms\n"
                  << "  Total promedio: " << sumTotal / processed << " ms\n"
                  << "  RAM maxima: " << peakRam << " MB\n";
        if (peakVram > 0)
            std::cout << "  VRAM maxima: " << peakVram << " MB\n";
        std::cout << "  Video: " << (o.save ? o.output : "no guardado") << "\n"
                  << "========================================\n";

        exportCsv(o, inputW, inputH, processed, avgFps,
                  sumInfer / processed, sumTotal / processed,
                  peakRam, peakVram, network);

        return 0;
    } catch (const cv::Exception& e) {
        std::cerr << "Error OpenCV: " << e.what() << "\n"; return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n"; return 1;
    }
}
