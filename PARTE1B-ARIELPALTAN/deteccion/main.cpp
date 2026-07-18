#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
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
    std::string model;
    std::string modelName;
    std::string input;
    std::string output;
    std::string device = "cpu";
    std::string classesFile = "config/classes.txt";
    int    imgsz      = 640;
    float  conf       = 0.25f;
    float  iou        = 0.45f;
    int    warmup     = 20;
    int    maxFrames  = 0;
    bool   show       = true;
    bool   save       = true;
};

struct LetterboxInfo {
    cv::Mat image;
    float   scale;
    int     padX, padY;
};

struct Detection {
    cv::Rect box;
    int      classId;
    float    confidence;
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

struct Times {
    double preprocess = 0, inference = 0, postprocess = 0;
    double drawing = 0, writing = 0, total = 0;
};

// ============================================================
//  Utilidades
// ============================================================
double elapsedMs(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

bool parseBool(const std::string& v) {
    if (v == "true" || v == "1" || v == "yes") return true;
    if (v == "false" || v == "0" || v == "no") return false;
    throw std::runtime_error("Valor booleano invalido: " + v);
}

Options parseArgs(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "--help" || key == "-h") {
            std::cout <<
                "Uso: ./yolo_bench [OPCIONES]\n"
                "\nRequerido:\n"
                "  --model RUTA           Modelo ONNX\n"
                "  --model-name NOMBRE    Nombre descriptivo\n"
                "  --source RUTA          Video de entrada\n"
                "\nOpcional:\n"
                "  --device cpu|cuda|cuda_fp16   (default: cpu)\n"
                "  --output RUTA          Video de salida\n"
                "  --classes RUTA         Archivo de clases (default: config/classes.txt)\n"
                "  --imgsz N              Resolucion de inferencia (default: 640)\n"
                "  --conf FLOAT           Umbral de confianza (default: 0.25)\n"
                "  --iou FLOAT            Umbral IoU NMS (default: 0.45)\n"
                "  --warmup N             Frames de calentamiento (default: 20)\n"
                "  --max-frames N         Max frames a procesar (0=todos)\n"
                "  --show true|false      Mostrar ventana (default: true)\n"
                "  --save true|false      Guardar video (default: true)\n";
            std::exit(0);
        }
        if (i + 1 >= argc) throw std::runtime_error("Falta valor para " + key);
        std::string val = argv[++i];
        if      (key == "--model")      o.model = val;
        else if (key == "--model-name") o.modelName = val;
        else if (key == "--source")     o.input = val;
        else if (key == "--output")     o.output = val;
        else if (key == "--device")     o.device = val;
        else if (key == "--classes")    o.classesFile = val;
        else if (key == "--imgsz")      o.imgsz = std::stoi(val);
        else if (key == "--conf")       o.conf = std::stof(val);
        else if (key == "--iou")        o.iou = std::stof(val);
        else if (key == "--warmup")     o.warmup = std::stoi(val);
        else if (key == "--max-frames") o.maxFrames = std::stoi(val);
        else if (key == "--show")       o.show = parseBool(val);
        else if (key == "--save")       o.save = parseBool(val);
        else throw std::runtime_error("Argumento desconocido: " + key);
    }
    if (o.model.empty()) throw std::runtime_error("--model es obligatorio");
    if (o.input.empty()) throw std::runtime_error("--source es obligatorio");
    if (o.modelName.empty()) o.modelName = fs::path(o.model).stem().string();
    if (o.device != "cpu" && o.device != "cuda" && o.device != "cuda_fp16")
        throw std::runtime_error("--device debe ser cpu, cuda o cuda_fp16");
    return o;
}

std::string deviceLabel(const std::string& d) {
    if (d == "cuda") return "GPU CUDA FP32";
    if (d == "cuda_fp16") return "GPU CUDA FP16";
    return "CPU";
}

std::vector<std::string> loadClasses(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "[WARN] No se pudo abrir " << path << ", usando etiquetas genericas\n";
        std::vector<std::string> v;
        for (int i = 0; i < 80; ++i) v.push_back("cls_" + std::to_string(i));
        return v;
    }
    std::vector<std::string> v;
    std::string line;
    while (std::getline(f, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == ' '))
            line.pop_back();
        if (!line.empty()) v.push_back(line);
    }
    return v;
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
//  Letterbox
// ============================================================
LetterboxInfo letterbox(const cv::Mat& frame, int size) {
    float scale = std::min(size / static_cast<float>(frame.cols),
                           size / static_cast<float>(frame.rows));
    int rw = std::max(1, static_cast<int>(std::round(frame.cols * scale)));
    int rh = std::max(1, static_cast<int>(std::round(frame.rows * scale)));
    int px = (size - rw) / 2, py = (size - rh) / 2;
    cv::Mat resized, out(size, size, CV_8UC3, cv::Scalar(114, 114, 114));
    cv::resize(frame, resized, cv::Size(rw, rh), 0, 0, cv::INTER_LINEAR);
    resized.copyTo(out(cv::Rect(px, py, rw, rh)));
    return {out, scale, px, py};
}

cv::Rect inverseLetterbox(const cv::Rect2f& box, const LetterboxInfo& lb, cv::Size orig, int imgsz) {
    float x1 = (box.x * imgsz - lb.padX) / lb.scale;
    float y1 = (box.y * imgsz - lb.padY) / lb.scale;
    float x2 = ((box.x + box.width)  * imgsz - lb.padX) / lb.scale;
    float y2 = ((box.y + box.height) * imgsz - lb.padY) / lb.scale;
    return cv::Rect(
        std::clamp(static_cast<int>(x1), 0, orig.width),
        std::clamp(static_cast<int>(y1), 0, orig.height),
        std::clamp(static_cast<int>(x2 - x1), 0, orig.width),
        std::clamp(static_cast<int>(y2 - y1), 0, orig.height)
    );
}

// ============================================================
//  NMS
// ============================================================
std::vector<Detection> applyNMS(std::vector<Detection>& dets, float iouThresh) {
    if (dets.empty()) return {};
    std::sort(dets.begin(), dets.end(),
              [](const Detection& a, const Detection& b) { return a.confidence > b.confidence; });

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    for (auto& d : dets) {
        int x = d.box.x;
        int y = d.box.y;
        int w = d.box.width;
        int h = d.box.height;
        int offset = d.classId * 10001;
        boxes.push_back(cv::Rect(x + offset, y + offset, w, h));
        scores.push_back(d.confidence);
    }
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, 0.f, iouThresh, indices);

    std::vector<Detection> kept;
    for (int idx : indices) kept.push_back(dets[idx]);
    return kept;
}

// ============================================================
//  Output parsing: detect format and extract detections
// ============================================================
struct OutputInfo {
    int numAnchors = 0;
    int numChannels = 0;
    bool channelsFirst = false;
    int  numClasses = 0;
    bool hasObjectness = false;
    bool isEnd2End = false;
};

OutputInfo inspectOutput(const cv::Mat& out, int numClasses) {
    OutputInfo info;
    if (out.dims != 3 || out.size[0] != 1) return info;

    int A = out.size[1], B = out.size[2];
    int expected_nc = 4 + numClasses;
    int expected_noc = 5 + numClasses;

    if (A == 6 && B > 1) {
        info = {B, A, true, numClasses, false, true};
    } else if (B == 6 && A > 1) {
        info = {A, B, false, numClasses, false, true};
    } else if (A == expected_nc && B > 1) {
        info = {B, A, true, numClasses, false, false};
    } else if (B == expected_nc && A > 1) {
        info = {A, B, false, numClasses, false, false};
    } else if (A == expected_noc && B > 1) {
        info = {B, A, true, numClasses, true, false};
    } else if (B == expected_noc && A > 1) {
        info = {A, B, false, numClasses, true, false};
    } else if (A > B) {
        info = {A, B, true, numClasses, false, false};
    } else {
        info = {B, A, false, numClasses, false, false};
    }
    return info;
}

float getVal(const float* data, const OutputInfo& info, int anchor, int channel) {
    if (info.channelsFirst)
        return data[channel * info.numAnchors + anchor];
    else
        return data[anchor * info.numChannels + channel];
}

std::vector<Detection> parseDetections(const cv::Mat& out, const OutputInfo& info,
                                        float confThresh, int imgsz) {
    std::vector<Detection> dets;
    if (info.numAnchors == 0) return dets;

    const float* data = reinterpret_cast<const float*>(out.data);
    int objStart = info.hasObjectness ? 5 : 4;

    for (int i = 0; i < info.numAnchors; ++i) {
        if (info.isEnd2End) {
            float x1 = getVal(data, info, i, 0);
            float y1 = getVal(data, info, i, 1);
            float x2 = getVal(data, info, i, 2);
            float y2 = getVal(data, info, i, 3);
            float score = getVal(data, info, i, 4);
            int   cid   = static_cast<int>(getVal(data, info, i, 5));
            if (score < confThresh || x2 <= x1 || y2 <= y1) continue;
            float normX = x1 / imgsz;
            float normY = y1 / imgsz;
            float normW = (x2 - x1) / imgsz;
            float normH = (y2 - y1) / imgsz;
            Detection d;
            d.box = cv::Rect(0, 0, 0, 0);
            d.classId = cid;
            d.confidence = score;
            d.box.x = static_cast<int>(normX * 10000);
            d.box.y = static_cast<int>(normY * 10000);
            d.box.width  = static_cast<int>(normW * 10000);
            d.box.height = static_cast<int>(normH * 10000);
            dets.push_back(d);
        } else {
            float rawX = getVal(data, info, i, 0);
            float rawY = getVal(data, info, i, 1);
            float rawW = getVal(data, info, i, 2);
            float rawH = getVal(data, info, i, 3);

            float objScore = info.hasObjectness ? getVal(data, info, i, 4) : 1.0f;

            float maxScore = -1e9f;
            int   maxClass = 0;
            for (int c = 0; c < info.numClasses; ++c) {
                float s = getVal(data, info, i, objStart + c) * objScore;
                if (s > maxScore) { maxScore = s; maxClass = c; }
            }
            if (maxScore < confThresh) continue;

            float normX, normY, normW, normH;
            if (std::abs(rawX) > 2.0f || std::abs(rawY) > 2.0f) {
                if (rawX < 0 && rawY < 0) {
                    normX = (rawX + rawW / 2.0f) / imgsz;
                    normY = (rawY + rawH / 2.0f) / imgsz;
                    normW = rawW / imgsz;
                    normH = rawH / imgsz;
                } else {
                    normX = rawX / imgsz;
                    normY = rawY / imgsz;
                    normW = rawW / imgsz;
                    normH = rawH / imgsz;
                }
            } else {
                if (rawX > rawW / 2.0f && rawY > rawH / 2.0f) {
                    normX = (rawX - rawW / 2.0f) / imgsz;
                    normY = (rawY - rawH / 2.0f) / imgsz;
                    normW = rawW / imgsz;
                    normH = rawH / imgsz;
                } else {
                    normX = rawX;
                    normY = rawY;
                    normW = rawW;
                    normH = rawH;
                }
            }

            Detection d;
            d.box = cv::Rect(0, 0, 0, 0);
            d.classId = maxClass;
            d.confidence = maxScore;
            d.box.x = static_cast<int>(normX * 10000);
            d.box.y = static_cast<int>(normY * 10000);
            d.box.width  = static_cast<int>(normW * 10000);
            d.box.height = static_cast<int>(normH * 10000);
            dets.push_back(d);
        }
    }
    return dets;
}

// ============================================================
//  Drawing
// ============================================================
static const int kPalette[][3] = {
    {255,56,56},{255,157,151},{255,112,31},{255,178,29},{207,210,49},
    {72,249,10},{146,204,23},{61,219,134},{26,147,52},{0,212,187},
    {44,153,168},{0,194,255},{52,69,147},{100,115,255},{0,24,236},
    {132,56,255},{82,0,133},{203,56,255},{255,149,200},{255,55,199}
};

cv::Scalar classColor(int id) {
    auto& c = kPalette[id % 20];
    return cv::Scalar(c[2], c[1], c[0]);
}

void drawDetections(cv::Mat& frame, const std::vector<Detection>& dets,
                    const std::vector<std::string>& classes) {
    for (const auto& d : dets) {
        cv::Scalar color = classColor(d.classId);
        cv::rectangle(frame, d.box, color, 2);

        std::string label;
        if (d.classId < static_cast<int>(classes.size()))
            label = classes[d.classId];
        else
            label = "cls_" + std::to_string(d.classId);
        label += " " + std::to_string(static_cast<int>(d.confidence * 100)) + "%";

        int bl = 0;
        cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &bl);
        cv::Point tl(d.box.x, std::max(d.box.y - 1, ts.height));
        cv::rectangle(frame,
                      cv::Point(tl.x, tl.y - ts.height - bl),
                      cv::Point(tl.x + ts.width, tl.y + bl),
                      color, cv::FILLED);
        cv::putText(frame, label, tl, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }
}

void drawOverlay(cv::Mat& frame, const std::string& model, const std::string& device,
                 double fps, long ramMb, const std::string& mac) {
    std::ostringstream ssFps, ssRam;
    ssFps << std::fixed << std::setprecision(1) << "FPS: " << fps;
    ssRam << "RAM: " << ramMb << " MB";

    std::vector<std::pair<std::string, cv::Scalar>> lines = {
        {model, {100, 220, 255}},
        {device, {100, 255, 150}},
        {ssFps.str(), {255, 220, 80}},
        {ssRam.str(), {200, 200, 200}},
        {"MAC: " + mac, {180, 180, 255}},
    };

    double scale = std::max(0.55, frame.cols / 800.0);
    int thick = std::max(1, static_cast<int>(scale * 1.5));
    int dy = static_cast<int>(28 * scale);
    int margin = static_cast<int>(8 * scale);
    int pad = static_cast<int>(5 * scale);

    int maxW = 0;
    for (auto& [txt, _] : lines) {
        int bl = 0;
        auto sz = cv::getTextSize(txt, cv::FONT_HERSHEY_SIMPLEX, scale, thick, &bl);
        maxW = std::max(maxW, sz.width);
    }

    int boxH = static_cast<int>(lines.size()) * dy + pad * 2;
    cv::Mat roi = frame(cv::Rect(margin, margin,
                                  std::min(maxW + pad * 2, frame.cols - margin * 2),
                                  std::min(boxH, frame.rows - margin * 2)));
    cv::Mat dark(roi.size(), roi.type(), cv::Scalar(0, 0, 0));
    cv::addWeighted(roi, 0.35, dark, 0.65, 0, roi);

    int y = margin + pad + dy - static_cast<int>(3 * scale);
    for (auto& [txt, col] : lines) {
        cv::putText(frame, txt, cv::Point(margin + pad, y),
                    cv::FONT_HERSHEY_SIMPLEX, scale, col, thick, cv::LINE_AA);
        y += dy;
    }
}

// ============================================================
//  CSV export
// ============================================================
void exportCsv(const Options& o, cv::Size size, int frames, double avgFps,
               double minFps, double maxFps, const Times& avg,
               long peakRam, long peakVram, long totalDet,
               const std::string& mac, const std::string& gpuName) {
    fs::create_directories("resultados");
    const fs::path path("resultados/yolo_benchmark.csv");
    bool header = !fs::exists(path) || fs::file_size(path) == 0;
    std::ofstream f(path, std::ios::app);

    if (header)
        f << "timestamp,model_name,model_path,device,video,input_width,input_height,"
             "imgsz,frames,average_fps,min_fps,max_fps,average_preprocess_ms,"
             "average_inference_ms,average_postprocess_ms,average_total_ms,"
             "peak_ram_mb,peak_vram_mb,total_detections,gpu_name,mac_address\n";

    const std::time_t now = std::time(nullptr);
    std::tm tm{};
    localtime_r(&now, &tm);

    f << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S") << ','
      << o.modelName << ',' << o.model << ',' << deviceLabel(o.device) << ','
      << o.input << ',' << size.width << ',' << size.height << ','
      << o.imgsz << ',' << frames << ',' << std::fixed << std::setprecision(2)
      << avgFps << ',' << minFps << ',' << maxFps << ','
      << avg.preprocess << ',' << avg.inference << ',' << avg.postprocess << ','
      << avg.total << ',' << peakRam << ',' << peakVram << ','
      << totalDet << ',' << gpuName << ',' << mac << '\n';
    std::cout << "[INFO] CSV: " << path << "\n";
}

// ============================================================
//  Summary
// ============================================================
void printSummary(const Options& o, int processed, const Times& avg,
                  double avgFps, double minFps, double maxFps,
                  long peakRam, long peakVram) {
    std::cout << "\n========================================\n"
              << "  RESUMEN DETECCION YOLO\n"
              << "========================================\n"
              << "  Modelo  : " << o.modelName << "\n"
              << "  Device  : " << deviceLabel(o.device) << "\n"
              << "  Frames  : " << processed << "\n"
              << "----------------------------------------\n"
              << std::fixed << std::setprecision(2)
              << "  Preprocesamiento promedio : " << avg.preprocess << " ms\n"
              << "  Inferencia promedio       : " << avg.inference << " ms\n"
              << "  Postprocesamiento promedio: " << avg.postprocess << " ms\n"
              << "  Tiempo total promedio     : " << avg.total << " ms\n"
              << "----------------------------------------\n"
              << "  FPS promedio : " << avgFps << "\n"
              << "  FPS minimo   : " << minFps << "\n"
              << "  FPS maximo   : " << maxFps << "\n"
              << "  RAM maxima   : " << peakRam << " MB\n";
    if (peakVram > 0)
        std::cout << "  VRAM maxima  : " << peakVram << " MB\n";
    std::cout << "========================================\n";
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
        std::vector<std::string> classes = loadClasses(o.classesFile);
        std::string gpuName;

        if (o.device != "cpu") {
            int count = cv::cuda::getCudaEnabledDeviceCount();
            if (count <= 0)
                throw std::runtime_error("CUDA no disponible en OpenCV");
            cv::cuda::DeviceInfo info(0);
            gpuName = info.name();
        }

        cv::dnn::Net net;
        try { net = cv::dnn::readNetFromONNX(o.model); }
        catch (const cv::Exception& e) {
            throw std::runtime_error("Error al cargar ONNX: " + std::string(e.what()));
        }
        if (net.empty()) throw std::runtime_error("ONNX vacio");

        if (o.device == "cpu") {
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        } else {
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            net.setPreferableTarget(o.device == "cuda_fp16"
                                    ? cv::dnn::DNN_TARGET_CUDA_FP16
                                    : cv::dnn::DNN_TARGET_CUDA);
            net.enableFusion(false);
        }

        const std::vector<std::string> outNames = net.getUnconnectedOutLayersNames();
        if (outNames.empty()) throw std::runtime_error("Sin capas de salida");

        cv::VideoCapture cap(o.input);
        if (!cap.isOpened()) throw std::runtime_error("No se pudo abrir: " + o.input);
        int origW = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int origH = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        double srcFps = cap.get(cv::CAP_PROP_FPS);
        if (!(srcFps > 0)) srcFps = 30.0;
        int reportedFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

        // Test inference
        cv::Mat first;
        if (!cap.read(first) || first.empty())
            throw std::runtime_error("Video sin frames validos");
        LetterboxInfo lb0 = letterbox(first, o.imgsz);
        cv::Mat blob0 = cv::dnn::blobFromImage(lb0.image, 1.0/255.0,
                                                cv::Size(o.imgsz, o.imgsz),
                                                cv::Scalar(), true, false, CV_32F);
        std::vector<cv::Mat> outs0;
        net.setInput(blob0);
        net.forward(outs0, outNames);

        OutputInfo outInfo = inspectOutput(outs0[0], static_cast<int>(classes.size()));
        if (outInfo.numAnchors == 0)
            throw std::runtime_error("Formato de salida no reconocido");
        std::cout << "Formato de salida: " << (outInfo.isEnd2End ? "END2END" : (outInfo.channelsFirst ? "RAW_CxN" : "RAW_NxC"))
                  << " [" << (outInfo.channelsFirst ? outInfo.numChannels : outInfo.numAnchors) << "x"
                  << (outInfo.channelsFirst ? outInfo.numAnchors : outInfo.numChannels) << "]\n";

        // Warmup
        for (int i = 0; i < o.warmup; ++i) {
            net.setInput(blob0);
            std::vector<cv::Mat> w; net.forward(w, outNames);
        }
        cap.set(cv::CAP_PROP_POS_FRAMES, 0);

        // Video writer
        cv::VideoWriter writer;
        if (o.save && !o.output.empty()) {
            writer.open(o.output, cv::VideoWriter::fourcc('m','p','4','v'),
                        srcFps, cv::Size(origW, origH));
            if (!writer.isOpened())
                writer.open(o.output, cv::VideoWriter::fourcc('a','v','c','1'),
                            srcFps, cv::Size(origW, origH));
        }

        // Info header
        std::cout << "========================================\n"
                  << "YOLO OBJECT DETECTION BENCHMARK\n"
                  << "========================================\n"
                  << "Modelo: " << o.modelName << " (" << o.model << ")\n"
                  << "Video: " << o.input << " [" << origW << "x" << origH
                  << " @ " << std::fixed << std::setprecision(1) << srcFps << " FPS]\n"
                  << "Frames reportados: " << reportedFrames << "\n"
                  << "Dispositivo: " << deviceLabel(o.device) << "\n"
                  << "GPU: " << (gpuName.empty() ? "N/A" : gpuName) << "\n"
                  << "MAC (" << network.interfaceName << "): " << network.mac << "\n"
                  << "OpenCV: " << CV_VERSION << "\n"
                  << "Resolucion modelo: " << o.imgsz << "x" << o.imgsz << "\n"
                  << "Confianza: " << o.conf << " | IoU: " << o.iou << "\n"
                  << "Clases: " << classes.size() << "\n"
                  << "========================================\n";

        // Benchmark loop
        std::vector<double> fpsValues;
        Times sums{};
        long peakRam = 0, peakVram = 0, totalDet = 0;
        int processed = 0;
        GpuStats gpu;
        bool stopped = false;

        while ((o.maxFrames == 0 || processed < o.maxFrames) && !stopped) {
            cv::Mat frame;
            if (!cap.read(frame) || frame.empty()) break;

            auto t0 = Clock::now();
            LetterboxInfo lb = letterbox(frame, o.imgsz);
            cv::Mat blob = cv::dnn::blobFromImage(lb.image, 1.0/255.0,
                                                   cv::Size(o.imgsz, o.imgsz),
                                                   cv::Scalar(), true, false, CV_32F);
            auto t1 = Clock::now();

            net.setInput(blob);
            std::vector<cv::Mat> outs;
            net.forward(outs, outNames);
            auto t2 = Clock::now();

            // Parse
            OutputInfo curInfo = inspectOutput(outs[0], static_cast<int>(classes.size()));
            std::vector<Detection> rawDets = parseDetections(outs[0], curInfo, o.conf, o.imgsz);

            // Convert coordinates and apply NMS
            std::vector<Detection> dets;
            if (curInfo.isEnd2End) {
                dets = rawDets;
            } else {
                dets = applyNMS(rawDets, o.iou);
            }

            // Inverse letterbox to original frame coords
            for (auto& d : dets) {
                cv::Rect2f normBox(
                    static_cast<float>(d.box.x) / 10000.0f,
                    static_cast<float>(d.box.y) / 10000.0f,
                    static_cast<float>(d.box.width)  / 10000.0f,
                    static_cast<float>(d.box.height) / 10000.0f
                );
                d.box = inverseLetterbox(normBox, lb, frame.size(), o.imgsz);
            }

            auto t3 = Clock::now();

            drawDetections(frame, dets, classes);
            auto t4 = Clock::now();

            double preprocessMs = elapsedMs(t0, t1);
            double inferenceMs  = elapsedMs(t1, t2);
            double postprocessMs = elapsedMs(t2, t3);
            double drawingMs    = elapsedMs(t3, t4);

            Times t;
            t.preprocess  = preprocessMs;
            t.inference   = inferenceMs;
            t.postprocess = postprocessMs;
            t.drawing     = drawingMs;

            double totalMs = preprocessMs + inferenceMs + postprocessMs + drawingMs;
            double currentFps = totalMs > 0 ? 1000.0 / totalMs : 0;
            fpsValues.push_back(currentFps);

            sums.preprocess  += preprocessMs;
            sums.inference   += inferenceMs;
            sums.postprocess += postprocessMs;
            sums.drawing     += drawingMs;
            sums.total       += totalMs;

            totalDet += static_cast<long>(dets.size());
            peakRam = std::max(peakRam, readRamMB());

            if (o.device != "cpu" && (processed == 0 || processed % 10 == 0)) {
                gpu = readGpuStats();
                if (gpu.valid) peakVram = std::max(peakVram, gpu.used);
            }

            drawOverlay(frame, o.modelName, deviceLabel(o.device),
                       currentFps, readRamMB(), network.mac);

            if (o.save && writer.isOpened()) writer.write(frame);

            if (o.show) {
                cv::Mat display = frame;
                if (frame.cols > 1280 || frame.rows > 720) {
                    double s = std::min(1280.0/frame.cols, 720.0/frame.rows);
                    cv::resize(frame, display, {}, s, s);
                }
                cv::imshow("YOLO Detection | " + o.modelName + " [" + deviceLabel(o.device) + "]", display);
                int key = cv::waitKey(1) & 0xff;
                if (key == 27 || key == 'q') stopped = true;
            }

            ++processed;
            if (processed % 10 == 0) {
                std::cout << "[" << deviceLabel(o.device) << "] Frame " << processed
                          << " | FPS: " << std::fixed << std::setprecision(1) << currentFps
                          << " | Infer: " << std::setprecision(2) << inferenceMs << " ms"
                          << " | Det: " << dets.size()
                          << " | RAM: " << readRamMB() << " MB";
                if (o.device != "cpu" && gpu.valid)
                    std::cout << " | VRAM: " << gpu.used << "/" << gpu.total << " MB";
                std::cout << "\n";
            }
        }

        cap.release();
        if (writer.isOpened()) writer.release();
        if (o.show) cv::destroyAllWindows();

        if (processed == 0) throw std::runtime_error("Sin frames procesados");

        Times avg = sums;
        avg.preprocess  /= processed;
        avg.inference   /= processed;
        avg.postprocess /= processed;
        avg.drawing     /= processed;
        avg.total       /= processed;

        double avgFps = std::accumulate(fpsValues.begin(), fpsValues.end(), 0.0)
                        / fpsValues.size();
        auto [minIt, maxIt] = std::minmax_element(fpsValues.begin(), fpsValues.end());

        printSummary(o, processed, avg, avgFps, *minIt, *maxIt, peakRam, peakVram);
        exportCsv(o, {origW, origH}, processed, avgFps, *minIt, *maxIt,
                  avg, peakRam, peakVram, totalDet, network.mac, gpuName);

        return 0;
    } catch (const cv::Exception& e) {
        std::cerr << "Error OpenCV: " << e.what() << "\n"; return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n"; return 1;
    }
}
