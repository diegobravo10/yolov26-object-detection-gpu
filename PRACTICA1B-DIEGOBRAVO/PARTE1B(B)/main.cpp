#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <chrono>
#include <cctype>
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

struct Options {
    std::string model = "weights/realesr-general-x4v3.onnx";
    std::string input;
    std::string output = "results/output.mp4";
    std::string device = "cpu";
    int tile = 128;
    int tilePad = 10;
    int maxFrames = 0;
    bool show = true;
    bool save = true;
};

struct NetworkInfo { std::string interfaceName = "No disponible"; std::string mac = "No disponible"; };
struct GpuStats { long usedMB = 0, totalMB = 0; int utilization = 0, temperature = 0; bool valid = false; };

static bool parseBool(const std::string& text) {
    std::string value = text;
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (value == "true" || value == "1" || value == "yes" || value == "on") return true;
    if (value == "false" || value == "0" || value == "no" || value == "off") return false;
    throw std::runtime_error("valor booleano inválido: " + text + " (use true o false)");
}

static int parseNonNegativeInt(const std::string& text, const std::string& name) {
    std::size_t used = 0;
    long value = 0;
    try { value = std::stol(text, &used); }
    catch (...) { throw std::runtime_error("valor inválido para " + name + ": " + text); }
    if (used != text.size() || value < 0 || value > std::numeric_limits<int>::max())
        throw std::runtime_error("valor inválido para " + name + ": " + text);
    return static_cast<int>(value);
}

static void printUsage(const char* program) {
    std::cout << "Uso: " << program
              << " --input video --output salida [--model modelo.onnx]"
                 " [--device cpu|cuda|cuda_fp16] [--tile 128] [--tile-pad 10]"
                 " [--max-frames 0] [--show true|false] [--save true|false]\n";
}

static Options parseArguments(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help" || key == "-h") { printUsage(argv[0]); std::exit(0); }
        if (i + 1 >= argc) throw std::runtime_error("falta un valor después de " + key);
        const std::string value = argv[++i];
        if (key == "--model") o.model = value;
        else if (key == "--input") o.input = value;
        else if (key == "--output") o.output = value;
        else if (key == "--device") o.device = value;
        else if (key == "--tile") o.tile = parseNonNegativeInt(value, key);
        else if (key == "--tile-pad") o.tilePad = parseNonNegativeInt(value, key);
        else if (key == "--max-frames") o.maxFrames = parseNonNegativeInt(value, key);
        else if (key == "--show") o.show = parseBool(value);
        else if (key == "--save") o.save = parseBool(value);
        else throw std::runtime_error("argumento desconocido: " + key);
    }
    if (o.input.empty()) throw std::runtime_error("debe especificar --input");
    if (o.save && o.output.empty()) throw std::runtime_error("debe especificar --output cuando --save es true");
    if (o.device != "cpu" && o.device != "cuda" && o.device != "cuda_fp16")
        throw std::runtime_error("--device debe ser cpu, cuda o cuda_fp16");
    if (o.tile <= 0) throw std::runtime_error("tile inválido: --tile debe ser mayor que cero");
    if (o.tilePad < 0 || o.tilePad > 4096) throw std::runtime_error("tile padding inválido");
    if (o.tile > 8192 || o.tile + 2LL * o.tilePad > 8192)
        throw std::runtime_error("tile inválido: tile más padding es excesivamente grande");
    return o;
}

static NetworkInfo getNetworkInfo() {
    const fs::path base("/sys/class/net");
    std::error_code ec;
    if (!fs::exists(base, ec)) return {};
    for (const auto& entry : fs::directory_iterator(base, fs::directory_options::skip_permission_denied, ec)) {
        const std::string name = entry.path().filename().string();
        if (name == "lo" || !fs::exists(entry.path() / "device", ec)) continue;
        std::ifstream file(entry.path() / "address");
        std::string mac;
        if (!(file >> mac) || mac.size() != 17 || mac == "00:00:00:00:00:00") continue;
        std::transform(mac.begin(), mac.end(), mac.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
        return {name, mac};
    }
    return {};
}

static long getRamMB() {
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            std::istringstream in(line.substr(6));
            long kb = 0; in >> kb;
            return kb / 1024;
        }
    }
    return 0;
}

static GpuStats getGpuStats() {
    GpuStats stats;
    FILE* pipe = popen("nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits 2>/dev/null", "r");
    if (!pipe) return stats;
    char buffer[256]{};
    std::string line;
    if (std::fgets(buffer, sizeof(buffer), pipe)) line = buffer;
    const int result = pclose(pipe);
    if (result != 0 || line.empty()) return stats;
    std::replace(line.begin(), line.end(), ',', ' ');
    std::istringstream in(line);
    if (in >> stats.usedMB >> stats.totalMB >> stats.utilization >> stats.temperature) stats.valid = true;
    return stats;
}

static std::string deviceLabel(const std::string& device) {
    if (device == "cuda") return "GPU CUDA FP32";
    if (device == "cuda_fp16") return "GPU CUDA FP16";
    return "CPU";
}

static cv::Mat tensorToBgr(const cv::Mat& output, int expectedWidth, int expectedHeight) {
    if (output.dims != 4) throw std::runtime_error("salida ONNX inesperada: se esperaban 4 dimensiones");
    if (output.size[0] != 1) throw std::runtime_error("salida ONNX inesperada: batch distinto de 1");
    if (output.size[1] != 3) throw std::runtime_error("salida ONNX inesperada: número de canales distinto de 3");
    const int h = output.size[2], w = output.size[3];
    if (h != expectedHeight || w != expectedWidth)
        throw std::runtime_error("salida ONNX inesperada: resolución " + std::to_string(w) + "x" + std::to_string(h) +
                                 ", esperada " + std::to_string(expectedWidth) + "x" + std::to_string(expectedHeight));
    cv::Mat contiguous = output.isContinuous() ? output : output.clone();
    std::vector<cv::Mat> rgbChannels;
    rgbChannels.reserve(3);
    for (int c = 0; c < 3; ++c)
        rgbChannels.emplace_back(h, w, CV_32F, contiguous.ptr<float>(0, c));
    cv::Mat rgbFloat, rgb8, bgr;
    cv::merge(rgbChannels, rgbFloat);
    cv::max(rgbFloat, 0.0, rgbFloat);
    cv::min(rgbFloat, 1.0, rgbFloat);
    rgbFloat.convertTo(rgb8, CV_8UC3, 255.0);
    cv::cvtColor(rgb8, bgr, cv::COLOR_RGB2BGR);
    return bgr;
}

static cv::Mat buildPaddedTile(const cv::Mat& frame, int x, int y, int coreW, int coreH, int tile, int pad) {
    const int wantedX0 = x - pad, wantedY0 = y - pad;
    const int wantedX1 = x + tile + pad, wantedY1 = y + tile + pad;
    const int sx0 = std::max(0, wantedX0), sy0 = std::max(0, wantedY0);
    const int sx1 = std::min(frame.cols, wantedX1), sy1 = std::min(frame.rows, wantedY1);
    if (coreW <= 0 || coreH <= 0 || sx1 <= sx0 || sy1 <= sy0) throw std::runtime_error("región de tile inválida");
    cv::Mat crop = frame(cv::Rect(sx0, sy0, sx1 - sx0, sy1 - sy0));
    cv::Mat padded;
    cv::copyMakeBorder(crop, padded, sy0 - wantedY0, wantedY1 - sy1,
                       sx0 - wantedX0, wantedX1 - sx1, cv::BORDER_REFLECT_101);
    return padded;
}

static cv::Mat upscaleFrame(cv::dnn::Net& net, const cv::Mat& frame, int tile, int pad, double& inferenceMs) {
    constexpr int scale = 4;
    cv::Mat result(frame.rows * scale, frame.cols * scale, CV_8UC3);
    inferenceMs = 0.0;
    for (int y = 0; y < frame.rows; y += tile) {
        const int coreH = std::min(tile, frame.rows - y);
        for (int x = 0; x < frame.cols; x += tile) {
            const int coreW = std::min(tile, frame.cols - x);
            cv::Mat padded = buildPaddedTile(frame, x, y, coreW, coreH, tile, pad);
            cv::Mat blob = cv::dnn::blobFromImage(padded, 1.0 / 255.0, cv::Size(), cv::Scalar(), true, false, CV_32F);
            net.setInput(blob);
            const auto start = Clock::now();
            cv::Mat output = net.forward();
            inferenceMs += std::chrono::duration<double, std::milli>(Clock::now() - start).count();
            cv::Mat tileBgr = tensorToBgr(output, padded.cols * scale, padded.rows * scale);
            const cv::Rect valid(pad * scale, pad * scale, coreW * scale, coreH * scale);
            const cv::Rect destination(x * scale, y * scale, coreW * scale, coreH * scale);
            tileBgr(valid).copyTo(result(destination));
        }
    }
    return result;
}

static void drawOverlay(cv::Mat& image, const std::string& device, double fps, double inferenceMs,
                        int frameNumber, int inputW, int inputH) {
    const std::vector<std::string> lines = {
        "Real-ESRGAN x4 | " + device,
        "FPS: " + cv::format("%.2f", fps) + " | Inferencia: " + cv::format("%.2f ms", inferenceMs),
        "Frame: " + std::to_string(frameNumber),
        "Entrada: " + std::to_string(inputW) + "x" + std::to_string(inputH) +
            " | Salida: " + std::to_string(image.cols) + "x" + std::to_string(image.rows)
    };
    const double fontScale = std::clamp(image.cols / 1600.0, 0.55, 1.1);
    const int thickness = std::max(1, static_cast<int>(fontScale * 2));
    int maxWidth = 0, baseline = 0;
    for (const auto& line : lines) maxWidth = std::max(maxWidth, cv::getTextSize(line, cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseline).width);
    const int lineHeight = static_cast<int>(30 * fontScale) + 8;
    cv::rectangle(image, cv::Rect(10, 10, std::min(maxWidth + 24, image.cols - 10),
                                  std::min(static_cast<int>(lines.size()) * lineHeight + 12, image.rows - 10)), cv::Scalar(0, 0, 0), cv::FILLED);
    for (std::size_t i = 0; i < lines.size(); ++i)
        cv::putText(image, lines[i], cv::Point(20, 10 + (static_cast<int>(i) + 1) * lineHeight - 8),
                    cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);
}

static cv::VideoWriter openWriter(const std::string& path, double fps, const cv::Size& size) {
    cv::VideoWriter writer;
    if (writer.open(path, cv::VideoWriter::fourcc('m','p','4','v'), fps, size)) return writer;
    std::cerr << "Advertencia: códec mp4v no disponible; intentando avc1.\n";
    if (writer.open(path, cv::VideoWriter::fourcc('a','v','c','1'), fps, size)) return writer;
    throw std::runtime_error("error de VideoWriter: no fue posible abrir mp4v ni avc1 para " + path);
}

int main(int argc, char** argv) {
    try {
        const Options options = parseArguments(argc, argv);
        if (!fs::is_regular_file(options.model)) throw std::runtime_error("modelo no encontrado: " + options.model);
        if (!fs::is_regular_file(options.input)) throw std::runtime_error("vídeo no encontrado: " + options.input);
        if (options.save) {
            const fs::path parent = fs::path(options.output).parent_path();
            if (!parent.empty()) fs::create_directories(parent);
        }

        const bool useCuda = options.device != "cpu";
        std::string gpuName;
        int cudaDevices = 0;
        if (useCuda) {
            cudaDevices = cv::cuda::getCudaEnabledDeviceCount();
            if (cudaDevices <= 0) throw std::runtime_error("CUDA no disponible: OpenCV no detectó dispositivos CUDA habilitados");
            cv::cuda::setDevice(0);
            cv::cuda::DeviceInfo info(0);
            if (!info.isCompatible()) throw std::runtime_error("CUDA no disponible: la GPU detectada no es compatible con esta compilación de OpenCV");
            gpuName = info.name();
        }

        cv::dnn::Net net;
        try { net = cv::dnn::readNetFromONNX(options.model); }
        catch (const cv::Exception& e) { throw std::runtime_error("error al abrir el ONNX: " + std::string(e.what())); }
        if (net.empty()) throw std::runtime_error("error al abrir el ONNX: la red está vacía");
        if (useCuda) {
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            net.setPreferableTarget(options.device == "cuda_fp16" ? cv::dnn::DNN_TARGET_CUDA_FP16 : cv::dnn::DNN_TARGET_CUDA);
        } else {
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        }

        cv::VideoCapture capture(options.input);
        if (!capture.isOpened()) throw std::runtime_error("error al abrir el vídeo: " + options.input);
        const int inputW = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
        const int inputH = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
        double sourceFps = capture.get(cv::CAP_PROP_FPS);
        const int reportedFrames = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_COUNT));
        if (inputW <= 0 || inputH <= 0) throw std::runtime_error("el vídeo tiene una resolución inválida");
        if (!std::isfinite(sourceFps) || sourceFps <= 0.0) throw std::runtime_error("el vídeo no contiene un FPS válido");
        const int expectedFrames = options.maxFrames > 0
            ? (reportedFrames > 0 ? std::min(reportedFrames, options.maxFrames) : options.maxFrames)
            : reportedFrames;
        const NetworkInfo network = getNetworkInfo();
        const std::string label = deviceLabel(options.device);

        cv::VideoWriter writer;
        if (options.save) writer = openWriter(options.output, sourceFps, cv::Size(inputW * 4, inputH * 4));

        std::cout << "========================================\n"
                  << "REAL-ESRGAN VIDEO SUPER RESOLUTION\n"
                  << "========================================\n"
                  << "Modelo: " << options.model << '\n'
                  << "Vídeo: " << options.input << '\n'
                  << "Salida: " << (options.save ? options.output : "desactivada") << '\n'
                  << "Dispositivo seleccionado: " << label << '\n'
                  << "OpenCV: " << CV_VERSION << '\n';
        if (useCuda) std::cout << "Dispositivos CUDA: " << cudaDevices << "\nGPU: " << gpuName << '\n';
        std::cout << "Interfaz de red: " << network.interfaceName << "\nMAC Address: " << network.mac << '\n'
                  << "Resolución de entrada: " << inputW << 'x' << inputH << '\n'
                  << "Resolución de salida: " << inputW * 4 << 'x' << inputH * 4 << '\n'
                  << "Escala: x4\nTile: " << options.tile << "\nTile padding: " << options.tilePad << '\n'
                  << "FPS original del vídeo: " << sourceFps << "\n========================================\n";

        const auto programStart = Clock::now();
        auto lastGpuUpdate = programStart;
        GpuStats gpuStats;
        double sumInference = 0.0, sumTotal = 0.0;
        long maxRam = 0, maxVram = 0;
        int processed = 0;
        bool stoppedByUser = false;
        cv::Mat frame;
        while ((options.maxFrames == 0 || processed < options.maxFrames) && capture.read(frame)) {
            const auto frameStart = Clock::now();
            double inferenceMs = 0.0;
            cv::Mat enhanced = upscaleFrame(net, frame, options.tile, options.tilePad, inferenceMs);
            const double elapsedBeforeOverlay = std::chrono::duration<double>(Clock::now() - frameStart).count();
            const double currentFps = elapsedBeforeOverlay > 0.0 ? 1.0 / elapsedBeforeOverlay : 0.0;
            ++processed;
            drawOverlay(enhanced, label, currentFps, inferenceMs, processed, inputW, inputH);
            if (options.save) writer.write(enhanced);
            if (options.show) {
                const double ratio = std::min({1.0, 1280.0 / enhanced.cols, 720.0 / enhanced.rows});
                cv::Mat preview;
                if (ratio < 1.0) cv::resize(enhanced, preview, cv::Size(), ratio, ratio, cv::INTER_AREA);
                else preview = enhanced;
                cv::imshow("Real-ESRGAN x4", preview);
                const int key = cv::waitKey(1) & 0xff;
                if (key == 27 || key == 'q' || key == 'Q') stoppedByUser = true;
            }
            const double totalMs = std::chrono::duration<double, std::milli>(Clock::now() - frameStart).count();
            sumInference += inferenceMs;
            sumTotal += totalMs;
            const long ram = getRamMB();
            maxRam = std::max(maxRam, ram);
            const auto now = Clock::now();
            if (useCuda && (processed == 1 || processed % 10 == 0 || now - lastGpuUpdate >= std::chrono::milliseconds(500))) {
                gpuStats = getGpuStats();
                lastGpuUpdate = now;
                if (gpuStats.valid) maxVram = std::max(maxVram, gpuStats.usedMB);
            }
            if (processed == 1 || processed % 5 == 0 || stoppedByUser || (expectedFrames > 0 && processed == expectedFrames)) {
                std::cout << '[' << label << "] Frame " << processed << '/'
                          << (expectedFrames > 0 ? std::to_string(expectedFrames) : "?")
                          << " | FPS: " << std::fixed << std::setprecision(2) << (totalMs > 0.0 ? 1000.0 / totalMs : 0.0)
                          << " | Inferencia: " << inferenceMs << " ms | Total: " << totalMs << " ms | RAM: " << ram << " MB";
                if (useCuda) {
                    if (gpuStats.valid) std::cout << " | VRAM: " << gpuStats.usedMB << '/' << gpuStats.totalMB
                                                  << " MB | GPU: " << gpuStats.utilization << "% | Temp: " << gpuStats.temperature << " C";
                    else std::cout << " | VRAM/GPU/Temp: no disponible";
                }
                std::cout << '\n';
            }
            if (stoppedByUser) break;
        }
        if (options.show) cv::destroyAllWindows();
        writer.release();
        const double totalSeconds = std::chrono::duration<double>(Clock::now() - programStart).count();
        const double avgFps = totalSeconds > 0.0 ? processed / totalSeconds : 0.0;
        std::cout << "========================================\nRESUMEN\n========================================\n"
                  << "Dispositivo: " << label << "\nFrames procesados: " << processed
                  << "\nFPS promedio: " << std::fixed << std::setprecision(2) << avgFps
                  << "\nInferencia promedio: " << (processed ? sumInference / processed : 0.0) << " ms"
                  << "\nTiempo total promedio: " << (processed ? sumTotal / processed : 0.0) << " ms"
                  << "\nTiempo total: " << totalSeconds << " s\nRAM máxima: " << maxRam << " MB";
        if (useCuda) std::cout << "\nVRAM máxima: " << maxVram << " MB";
        std::cout << "\nMAC Address: " << network.mac
                  << "\nVídeo generado: " << (options.save ? options.output : "guardado desactivado")
                  << "\n========================================\n";
        return 0;
    } catch (const cv::Exception& e) {
        const std::string message = e.what();
        if (message.find("OutOfMemory") != std::string::npos || message.find("out of memory") != std::string::npos)
            std::cerr << "Error de memoria CUDA/OpenCV: reduzca --tile. Detalle: " << message << '\n';
        else std::cerr << "Error de OpenCV: " << message << '\n';
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
    }
    return 1;
}
