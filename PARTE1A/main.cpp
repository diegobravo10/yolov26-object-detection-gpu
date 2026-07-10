#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
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

constexpr int kNumClasses = 3;
// YOLO26-seg exportado con end2end=false no incluye objectness. Cambiar a true
// solamente para una exportacion cuya fila sea [xywh, objectness, clases, masks].
constexpr bool kHasObjectness = false;
const std::vector<std::string> kClasses{"Ball", "Player", "Referee"};
const std::vector<cv::Scalar> kColors{{0, 255, 255}, {0, 255, 0}, {0, 0, 255}};

struct Options {
    std::string model = "models/yolo26_soccer_seg.onnx";
    std::string input = "partido.mp4";
    std::string output = "resultado.mp4";
    std::string device = "cpu";
    int imgsz = 640;
    float conf = 0.35F;
    float iou = 0.45F;
    float maskThreshold = 0.50F;
    int maxFrames = 0;
    bool show = true;
    bool save = true;
};

struct LetterboxInfo { cv::Mat image; float scale; int padX; int padY; int resizedW; int resizedH; };
struct OutputLayout { int detIndex = -1; int protoIndex = -1; int rows = 0; int cols = 0; bool channelsFirst = false; int maskDim = 0; int maskH = 0; int maskW = 0; };
struct Detection { cv::Rect box; int classId; float confidence; cv::Mat coefficients; cv::Mat mask; };
struct GpuStats { long used = 0; long total = 0; int utilization = 0; int temperature = 0; bool valid = false; };
struct NetworkInfo { std::string interfaceName = "No disponible"; std::string mac = "No disponible"; };
struct Times { double preprocess = 0, inference = 0, post = 0, drawing = 0, writing = 0, total = 0; };

double elapsedMs(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

bool parseBool(const std::string& value) {
    if (value == "true" || value == "1" || value == "yes") return true;
    if (value == "false" || value == "0" || value == "no") return false;
    throw std::runtime_error("Valor booleano invalido: " + value + " (use true o false)");
}

Options parseArgs(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help" || key == "-h") {
            std::cout << "Uso: ./yolo_seg [--model ruta] [--input video] [--output video] "
                         "[--device cpu|cuda|cuda_fp16] [--imgsz N] [--conf F] [--iou F] "
                         "[--mask-threshold F] [--max-frames N] [--show true|false] [--save true|false]\n";
            std::exit(0);
        }
        if (i + 1 >= argc) throw std::runtime_error("Falta valor para " + key);
        const std::string value = argv[++i];
        try {
            if (key == "--model") o.model = value;
            else if (key == "--input") o.input = value;
            else if (key == "--output") o.output = value;
            else if (key == "--device") o.device = value;
            else if (key == "--imgsz") o.imgsz = std::stoi(value);
            else if (key == "--conf") o.conf = std::stof(value);
            else if (key == "--iou") o.iou = std::stof(value);
            else if (key == "--mask-threshold") o.maskThreshold = std::stof(value);
            else if (key == "--max-frames") o.maxFrames = std::stoi(value);
            else if (key == "--show") o.show = parseBool(value);
            else if (key == "--save") o.save = parseBool(value);
            else throw std::runtime_error("Argumento desconocido: " + key);
        } catch (const std::invalid_argument&) { throw std::runtime_error("Valor invalido para " + key + ": " + value); }
        catch (const std::out_of_range&) { throw std::runtime_error("Valor fuera de rango para " + key + ": " + value); }
    }
    if (o.device != "cpu" && o.device != "cuda" && o.device != "cuda_fp16")
        throw std::runtime_error("--device debe ser cpu, cuda o cuda_fp16");
    if (o.imgsz <= 0 || o.maxFrames < 0 || o.conf < 0 || o.conf > 1 || o.iou < 0 || o.iou > 1 ||
        o.maskThreshold < 0 || o.maskThreshold > 1) throw std::runtime_error("Parametros numericos fuera de rango");
    return o;
}

std::string deviceLabel(const std::string& d) {
    if (d == "cuda") return "GPU CUDA FP32";
    if (d == "cuda_fp16") return "GPU CUDA FP16";
    return "CPU";
}

NetworkInfo getNetworkInfo() {
    NetworkInfo result;
    std::error_code ec;
    for (const auto& entry : fs::directory_iterator("/sys/class/net", ec)) {
        const std::string name = entry.path().filename().string();
        if (name == "lo") continue;
        const fs::path devicePath = entry.path() / "device";
        if (!fs::exists(devicePath, ec)) continue; // descarta interfaces virtuales
        std::ifstream f(entry.path() / "address");
        std::string mac;
        if (!(f >> mac) || mac == "00:00:00:00:00:00" || mac.size() != 17) continue;
        result = {name, mac};
        break;
    }
    return result;
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
    FILE* pipe = popen("nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits 2>/dev/null", "r");
    if (!pipe) return s;
    char buffer[256]{};
    if (std::fgets(buffer, sizeof(buffer), pipe)) {
        std::string line(buffer); std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream in(line);
        s.valid = static_cast<bool>(in >> s.used >> s.total >> s.utilization >> s.temperature);
    }
    pclose(pipe);
    return s;
}

LetterboxInfo letterbox(const cv::Mat& frame, int size) {
    const float scale = std::min(size / static_cast<float>(frame.cols), size / static_cast<float>(frame.rows));
    const int rw = std::max(1, static_cast<int>(std::round(frame.cols * scale)));
    const int rh = std::max(1, static_cast<int>(std::round(frame.rows * scale)));
    const int px = (size - rw) / 2, py = (size - rh) / 2;
    cv::Mat resized, out(size, size, CV_8UC3, cv::Scalar(114, 114, 114));
    cv::resize(frame, resized, cv::Size(rw, rh), 0, 0, cv::INTER_LINEAR);
    resized.copyTo(out(cv::Rect(px, py, rw, rh)));
    return {out, scale, px, py, rw, rh};
}

std::string shapeString(const cv::Mat& m) {
    std::ostringstream s; s << '[';
    for (int i = 0; i < m.dims; ++i) { if (i) s << ", "; s << m.size[i]; }
    return s.str() + ']';
}

OutputLayout inspectOutputs(const std::vector<cv::Mat>& outs, bool print) {
    OutputLayout layout;
    for (size_t i = 0; i < outs.size(); ++i) {
        if (print) std::cout << "Salida " << i << ": " << shapeString(outs[i]) << '\n';
        const cv::Mat& m = outs[i];
        if (m.dims == 4 && m.size[0] == 1 && m.size[1] > 0 && m.size[2] > 0 && m.size[3] > 0) {
            if (layout.protoIndex >= 0) throw std::runtime_error("Se encontraron varios tensores candidatos a prototipos");
            layout.protoIndex = static_cast<int>(i); layout.maskDim = m.size[1]; layout.maskH = m.size[2]; layout.maskW = m.size[3];
        }
    }
    if (layout.protoIndex < 0) throw std::runtime_error("Ausencia del tensor de prototipos [1, mask_dim, mask_h, mask_w]");
    const int expected = 4 + kNumClasses + layout.maskDim + (kHasObjectness ? 1 : 0);
    for (size_t i = 0; i < outs.size(); ++i) {
        const cv::Mat& m = outs[i];
        if (m.dims != 3 || m.size[0] != 1) continue;
        bool cf = m.size[1] == expected, cl = m.size[2] == expected;
        if (cf == cl) continue;
        if (layout.detIndex >= 0) throw std::runtime_error("Se encontraron varios tensores candidatos a detecciones");
        layout.detIndex = static_cast<int>(i); layout.channelsFirst = cf;
        layout.rows = cf ? m.size[2] : m.size[1]; layout.cols = expected;
    }
    if (layout.detIndex < 0) {
        std::ostringstream e;
        e << "Forma de salida no reconocida. Se esperaba deteccion [1," << expected << ",N] o [1,N," << expected
          << "] porque 4 caja + " << kNumClasses << " clases + " << layout.maskDim << " mascaras"
          << (kHasObjectness ? " + 1 objectness." : ". El parser esta configurado sin objectness.");
        throw std::runtime_error(e.str());
    }
    if (layout.maskDim <= 0 || layout.cols != expected) throw std::runtime_error("Incompatibilidad de dimensiones de mascara");
    return layout;
}

cv::Mat detectionRows(const cv::Mat& raw, const OutputLayout& l) {
    cv::Mat two(l.channelsFirst ? l.cols : l.rows, l.channelsFirst ? l.rows : l.cols, CV_32F,
                const_cast<float*>(raw.ptr<float>()));
    cv::Mat rows = l.channelsFirst ? two.t() : two;
    return rows.isContinuous() ? rows : rows.clone();
}

std::vector<Detection> postprocess(const std::vector<cv::Mat>& outs, const OutputLayout& l,
                                   const LetterboxInfo& lb, const cv::Size& original, const Options& o) {
    cv::Mat rows = detectionRows(outs[l.detIndex], l);
    std::vector<cv::Rect> boxes; std::vector<float> scores; std::vector<int> classIds; std::vector<cv::Mat> coeffs;
    const int classStart = 4 + (kHasObjectness ? 1 : 0);
    const int coeffStart = classStart + kNumClasses;
    for (int r = 0; r < rows.rows; ++r) {
        const float* p = rows.ptr<float>(r);
        int cid = 0; float classScore = p[classStart];
        for (int c = 1; c < kNumClasses; ++c) if (p[classStart + c] > classScore) { classScore = p[classStart + c]; cid = c; }
        const float score = classScore * (kHasObjectness ? p[4] : 1.0F);
        if (!std::isfinite(score) || score < o.conf) continue;
        float x1 = (p[0] - p[2] * 0.5F - lb.padX) / lb.scale;
        float y1 = (p[1] - p[3] * 0.5F - lb.padY) / lb.scale;
        float x2 = (p[0] + p[2] * 0.5F - lb.padX) / lb.scale;
        float y2 = (p[1] + p[3] * 0.5F - lb.padY) / lb.scale;
        int left = std::clamp(static_cast<int>(std::floor(x1)), 0, original.width);
        int top = std::clamp(static_cast<int>(std::floor(y1)), 0, original.height);
        int right = std::clamp(static_cast<int>(std::ceil(x2)), 0, original.width);
        int bottom = std::clamp(static_cast<int>(std::ceil(y2)), 0, original.height);
        if (right <= left || bottom <= top) continue;
        boxes.emplace_back(left, top, right - left, bottom - top); scores.push_back(score); classIds.push_back(cid);
        coeffs.emplace_back(1, l.maskDim, CV_32F); std::copy(p + coeffStart, p + coeffStart + l.maskDim, coeffs.back().ptr<float>());
    }
    // NMS por clase: cajas de clases diferentes no deben suprimirse entre si.
    std::vector<int> keep;
    for (int cls = 0; cls < kNumClasses; ++cls) {
        std::vector<cv::Rect> b; std::vector<float> s; std::vector<int> map, idx;
        for (size_t i = 0; i < boxes.size(); ++i) if (classIds[i] == cls) { b.push_back(boxes[i]); s.push_back(scores[i]); map.push_back(static_cast<int>(i)); }
        cv::dnn::NMSBoxes(b, s, o.conf, o.iou, idx);
        for (int i : idx) keep.push_back(map[i]);
    }
    const cv::Mat& proto4 = outs[l.protoIndex];
    cv::Mat prototypes(l.maskDim, l.maskH * l.maskW, CV_32F, const_cast<float*>(proto4.ptr<float>()));
    if (prototypes.rows != l.maskDim || prototypes.cols != l.maskH * l.maskW)
        throw std::runtime_error("Incompatibilidad al convertir los prototipos a matriz");
    std::vector<Detection> detections; detections.reserve(keep.size());
    const cv::Rect validInput(lb.padX, lb.padY, lb.resizedW, lb.resizedH);
    if (validInput.width <= 0 || validInput.height <= 0) throw std::runtime_error("ROI de letterbox vacia");
    for (int i : keep) {
        if (coeffs[i].cols != prototypes.rows) throw std::runtime_error("coeficientes x prototipos: dimensiones incompatibles");
        cv::Mat logits = coeffs[i] * prototypes;
        cv::Mat neg, expv, sigmoid; cv::multiply(logits, -1, neg); cv::exp(neg, expv); cv::divide(1.0, 1.0 + expv, sigmoid);
        cv::Mat small(l.maskH, l.maskW, CV_32F, sigmoid.ptr<float>()), inputMask;
        cv::resize(small, inputMask, cv::Size(o.imgsz, o.imgsz), 0, 0, cv::INTER_LINEAR);
        cv::Mat unpadded = inputMask(validInput), fullMask;
        cv::resize(unpadded, fullMask, original, 0, 0, cv::INTER_LINEAR);
        cv::Mat binary = cv::Mat::zeros(original, CV_8U);
        const cv::Rect box = boxes[i] & cv::Rect(0, 0, original.width, original.height);
        if (box.empty()) continue;
        cv::compare(fullMask(box), o.maskThreshold, binary(box), cv::CMP_GT);
        detections.push_back({box, classIds[i], scores[i], coeffs[i], binary});
    }
    return detections;
}

void drawDetections(cv::Mat& frame, const std::vector<Detection>& detections) {
    for (const auto& d : detections) {
        cv::Mat color(frame.size(), frame.type(), kColors[d.classId]);
        cv::Mat blended; cv::addWeighted(frame, 0.55, color, 0.45, 0, blended);
        blended.copyTo(frame, d.mask);
        cv::rectangle(frame, d.box, kColors[d.classId], 2);
        std::ostringstream label; label << kClasses[d.classId] << ' ' << std::fixed << std::setprecision(2) << d.confidence;
        int baseline = 0; cv::Size ts = cv::getTextSize(label.str(), cv::FONT_HERSHEY_SIMPLEX, 0.55, 1, &baseline);
        int y = std::max(d.box.y, ts.height + 6);
        cv::rectangle(frame, cv::Rect(d.box.x, y - ts.height - 6, std::min(ts.width + 6, frame.cols - d.box.x), ts.height + 6), kColors[d.classId], cv::FILLED);
        cv::putText(frame, label.str(), cv::Point(d.box.x + 3, y - 3), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    }
}

void drawOverlay(cv::Mat& frame, const std::string& device, int frameNo, int detections,
                 double fps, double avgFps, double inference,
                 const std::string& interfaceName, const std::string& mac) {
    std::vector<std::string> lines{"YOLO26 Soccer Segmentation", device};
    std::ostringstream a, b, c;
    a << std::fixed << std::setprecision(2) << "FPS: " << fps << "  Promedio: " << avgFps;
    b << std::fixed << std::setprecision(2) << "Inferencia: " << inference << " ms  Detecciones: " << detections;
    c << "Frame: " << frameNo; lines.push_back(a.str()); lines.push_back(b.str()); lines.push_back(c.str());
    lines.push_back("MAC (" + interfaceName + "): " + mac);
    const int width = std::min(470, frame.cols), height = std::min(157, frame.rows);
    cv::Mat roi = frame(cv::Rect(0, 0, width, height)), dark; roi.convertTo(dark, -1, 0.45, 0); dark.copyTo(roi);
    int y = 24; for (const auto& line : lines) { cv::putText(frame, line, {10, y}, cv::FONT_HERSHEY_SIMPLEX, 0.58, cv::Scalar(255,255,255), 1, cv::LINE_AA); y += 25; }
}

cv::VideoWriter openWriter(const std::string& path, double fps, cv::Size size) {
    fs::path p(path); if (p.has_parent_path()) fs::create_directories(p.parent_path());
    cv::VideoWriter writer(path, cv::VideoWriter::fourcc('m','p','4','v'), fps, size);
    if (!writer.isOpened()) writer.open(path, cv::VideoWriter::fourcc('a','v','c','1'), fps, size);
    if (!writer.isOpened()) throw std::runtime_error("VideoWriter no disponible para: " + path);
    return writer;
}

void appendCsv(const Options& o, cv::Size size, int frames, double avgFps, double minFps, double maxFps,
               const Times& avg, long peakRam, long peakVram, long detections, const std::string& mac) {
    const fs::path path("resultados_segmentacion.csv"); const bool header = !fs::exists(path) || fs::file_size(path) == 0;
    std::ofstream f(path, std::ios::app); if (!f) throw std::runtime_error("No se pudo abrir resultados_segmentacion.csv");
    if (header) f << "timestamp,device,model,input_video,input_width,input_height,imgsz,frames,average_fps,min_fps,max_fps,average_preprocess_ms,average_inference_ms,average_postprocess_ms,average_total_ms,peak_ram_mb,peak_vram_mb,total_detections,mac_address\n";
    const std::time_t now = std::time(nullptr); std::tm tm{}; localtime_r(&now, &tm);
    f << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S%z") << ',' << deviceLabel(o.device) << ',' << o.model << ',' << o.input << ','
      << size.width << ',' << size.height << ',' << o.imgsz << ',' << frames << ',' << std::fixed << std::setprecision(2)
      << avgFps << ',' << minFps << ',' << maxFps << ',' << avg.preprocess << ',' << avg.inference << ',' << avg.post << ','
      << avg.total << ',' << peakRam << ',' << peakVram << ',' << detections << ',' << mac << '\n';
}

int main(int argc, char** argv) {
    try {
        const Options o = parseArgs(argc, argv);
        if (!fs::is_regular_file(o.model)) throw std::runtime_error("Modelo inexistente: " + o.model);
        if (!fs::is_regular_file(o.input)) throw std::runtime_error("Video inexistente: " + o.input);
        cv::VideoCapture cap(o.input); if (!cap.isOpened()) throw std::runtime_error("No se pudo abrir el video: " + o.input);
        const int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        const int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        double sourceFps = cap.get(cv::CAP_PROP_FPS); if (!(sourceFps > 0)) sourceFps = 30.0;
        const int reportedFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
        const NetworkInfo network = getNetworkInfo();
        std::string gpuName;
        if (o.device != "cpu") {
            const int count = cv::cuda::getCudaEnabledDeviceCount();
            std::cout << "Dispositivos CUDA disponibles: " << count << '\n';
            if (count <= 0) throw std::runtime_error("CUDA no disponible o OpenCV fue compilado sin backend CUDA");
            cv::cuda::DeviceInfo info(0); gpuName = info.name();
        }
        cv::dnn::Net net;
        try { net = cv::dnn::readNetFromONNX(o.model); }
        catch (const cv::Exception& e) { throw std::runtime_error(std::string("ONNX invalido o no compatible: ") + e.what()); }
        if (net.empty()) throw std::runtime_error("ONNX invalido: la red esta vacia");
        if (o.device == "cpu") { net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV); net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU); }
        else {
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            net.setPreferableTarget(o.device == "cuda" ? cv::dnn::DNN_TARGET_CUDA : cv::dnn::DNN_TARGET_CUDA_FP16);
            // Workaround para una asercion del fusionador de OpenCV 4.11 con
            // ciertos bloques Conv+Add de YOLO26. El backend sigue siendo CUDA.
            net.enableFusion(false);
        }
        const std::vector<std::string> outputNames = net.getUnconnectedOutLayersNames();
        if (outputNames.empty()) throw std::runtime_error("La red ONNX no tiene salidas desconectadas");
        cv::Mat first; if (!cap.read(first) || first.empty()) throw std::runtime_error("El video no contiene un primer frame valido");
        LetterboxInfo initialLb = letterbox(first, o.imgsz);
        cv::Mat initialBlob = cv::dnn::blobFromImage(initialLb.image, 1.0 / 255.0, cv::Size(o.imgsz, o.imgsz), cv::Scalar(), true, false, CV_32F);
        std::vector<cv::Mat> initialOuts;
        try { net.setInput(initialBlob); net.forward(initialOuts, outputNames); }
        catch (const cv::Exception& e) {
            if (o.device != "cpu") throw std::runtime_error(std::string("Fallo la prueba del backend CUDA; no se cambiara a CPU: ") + e.what());
            throw;
        }
        std::cout << "Nombres y dimensiones de las salidas ONNX:\n";
        for (size_t i = 0; i < initialOuts.size(); ++i) {
            const std::string name = i < outputNames.size() ? outputNames[i] : "sin_nombre";
            std::cout << "Salida " << i << " (" << name << "): " << shapeString(initialOuts[i]) << '\n';
        }
        const OutputLayout layout = inspectOutputs(initialOuts, false);
        (void)postprocess(initialOuts, layout, initialLb, first.size(), o);
        std::cout << "Prueba inicial ONNX correcta.\n";

        std::cout << "========================================\nYOLO26 SOCCER SEGMENTATION\n========================================\n"
                  << "Modelo: " << o.model << "\nVideo: " << o.input << "\nSalida: " << (o.save ? o.output : "desactivada")
                  << "\nDispositivo: " << deviceLabel(o.device) << "\nOpenCV: " << CV_VERSION;
        if (!gpuName.empty()) std::cout << "\nGPU: " << gpuName;
        std::cout << "\nInterfaz de red: " << network.interfaceName << "\nMAC Address: " << network.mac
                  << "\nResolucion de video: " << width << 'x' << height << "\nEntrada del modelo: " << o.imgsz << 'x' << o.imgsz
                  << "\nFPS original: " << sourceFps << "\nClases: Ball, Player, Referee\nConfianza: " << o.conf
                  << "\nIoU NMS: " << o.iou << "\nUmbral de mascara: " << o.maskThreshold << "\n========================================\n";

        // Cinco inferencias de calentamiento, independientes del benchmark y del video generado.
        for (int i = 0; i < 5; ++i) { net.setInput(initialBlob); std::vector<cv::Mat> warm; net.forward(warm, outputNames); }
        cap.set(cv::CAP_PROP_POS_FRAMES, 0);
        cv::VideoWriter writer; if (o.save) writer = openWriter(o.output, sourceFps, {width, height});
        std::vector<double> fpsValues; Times sums; long peakRam = 0, peakVram = 0, totalDetections = 0;
        int processed = 0; GpuStats gpu;
        while (o.maxFrames == 0 || processed < o.maxFrames) {
            cv::Mat frame; if (!cap.read(frame) || frame.empty()) break;
            const auto totalStart = Clock::now();
            const auto preStart = totalStart; LetterboxInfo lb = letterbox(frame, o.imgsz);
            cv::Mat blob = cv::dnn::blobFromImage(lb.image, 1.0 / 255.0, cv::Size(o.imgsz, o.imgsz), cv::Scalar(), true, false, CV_32F);
            const auto preEnd = Clock::now(); net.setInput(blob); std::vector<cv::Mat> outs; net.forward(outs, outputNames); const auto inferEnd = Clock::now();
            OutputLayout current = inspectOutputs(outs, false);
            if (current.detIndex != layout.detIndex || current.protoIndex != layout.protoIndex || current.maskDim != layout.maskDim)
                throw std::runtime_error("Las dimensiones de salida cambiaron durante el video");
            auto detections = postprocess(outs, current, lb, frame.size(), o); const auto postEnd = Clock::now();
            drawDetections(frame, detections); const auto drawEnd = Clock::now();
            // Total provisional excluye waitKey; el overlay se calcula con el tiempo acumulado hasta dibujo.
            double provisional = elapsedMs(totalStart, drawEnd);
            double currentFps = provisional > 0 ? 1000.0 / provisional : 0;
            double previousSum = std::accumulate(fpsValues.begin(), fpsValues.end(), 0.0);
            drawOverlay(frame, deviceLabel(o.device), processed + 1, static_cast<int>(detections.size()), currentFps,
                        fpsValues.empty() ? currentFps : previousSum / fpsValues.size(), elapsedMs(preEnd, inferEnd),
                        network.interfaceName, network.mac);
            const auto writeStart = Clock::now(); if (o.save) writer.write(frame); const auto writeEnd = Clock::now();
            Times t; t.preprocess = elapsedMs(preStart, preEnd); t.inference = elapsedMs(preEnd, inferEnd); t.post = elapsedMs(inferEnd, postEnd);
            t.drawing = elapsedMs(postEnd, drawEnd); t.writing = elapsedMs(writeStart, writeEnd); t.total = elapsedMs(totalStart, writeEnd);
            currentFps = t.total > 0 ? 1000.0 / t.total : 0; fpsValues.push_back(currentFps); ++processed;
            sums.preprocess += t.preprocess; sums.inference += t.inference; sums.post += t.post; sums.drawing += t.drawing; sums.writing += t.writing; sums.total += t.total;
            totalDetections += static_cast<long>(detections.size()); peakRam = std::max(peakRam, readRamMB());
            if (o.device != "cpu" && (processed == 1 || processed % 10 == 0)) { gpu = readGpuStats(); if (gpu.valid) peakVram = std::max(peakVram, gpu.used); }
            if (processed % 5 == 0) {
                const double avgFps = std::accumulate(fpsValues.begin(), fpsValues.end(), 0.0) / fpsValues.size();
                std::cout << '[' << deviceLabel(o.device) << "] Frame " << processed << '/'
                          << (o.maxFrames > 0 ? std::min(o.maxFrames, reportedFrames) : reportedFrames)
                          << " | FPS actual: " << std::fixed << std::setprecision(2) << currentFps << " | FPS promedio: " << avgFps
                          << " | Inferencia: " << t.inference << " ms | Post: " << t.post << " ms | Total: " << t.total
                          << " ms | Detecciones: " << detections.size() << " | RAM: " << readRamMB() << " MB";
                if (o.device != "cpu" && gpu.valid) std::cout << " | VRAM: " << gpu.used << '/' << gpu.total << " MB | GPU: " << gpu.utilization << "% | Temp: " << gpu.temperature << " C";
                std::cout << '\n';
            }
            if (o.show) {
                cv::Mat display = frame; if (frame.cols > 1280 || frame.rows > 720) { double s = std::min(1280.0/frame.cols, 720.0/frame.rows); cv::resize(frame, display, {}, s, s); }
                cv::imshow("YOLO26 Soccer Segmentation", display); int key = cv::waitKey(1) & 0xff; if (key == 27 || key == 'q') break;
            }
        }
        cap.release(); if (writer.isOpened()) writer.release(); if (o.show) cv::destroyAllWindows();
        if (processed == 0) throw std::runtime_error("No se proceso ningun frame");
        Times avg = sums; avg.preprocess /= processed; avg.inference /= processed; avg.post /= processed; avg.drawing /= processed; avg.writing /= processed; avg.total /= processed;
        const double avgFps = std::accumulate(fpsValues.begin(), fpsValues.end(), 0.0) / fpsValues.size();
        const auto [minIt, maxIt] = std::minmax_element(fpsValues.begin(), fpsValues.end());
        std::cout << "========================================\nRESUMEN DE SEGMENTACION\n========================================\nDispositivo: " << deviceLabel(o.device)
                  << "\nFrames procesados: " << processed << "\nFrames medidos: " << fpsValues.size() << "\nFPS promedio: " << std::fixed << std::setprecision(2) << avgFps
                  << "\nFPS minimo: " << *minIt << "\nFPS maximo: " << *maxIt << "\nPreprocesamiento promedio: " << avg.preprocess
                  << " ms\nInferencia promedio: " << avg.inference << " ms\nPostprocesamiento promedio: " << avg.post << " ms\nTiempo total promedio: " << avg.total
                  << " ms\nRAM maxima: " << peakRam << " MB\nVRAM maxima: " << peakVram << " MB\nDetecciones totales: " << totalDetections
                  << "\nMAC Address: " << network.mac << "\nVideo generado: " << (o.save ? o.output : "no (--save false)") << "\n========================================\n";
        appendCsv(o, {width, height}, processed, avgFps, *minIt, *maxIt, avg, peakRam, peakVram, totalDetections, network.mac);
        return 0;
    } catch (const cv::Exception& e) {
        std::cerr << "Error de OpenCV: " << e.what() << '\n'; return 1;
    } catch (const std::bad_alloc& e) {
        std::cerr << "Error de memoria: " << e.what() << '\n'; return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n'; return 1;
    }
}
