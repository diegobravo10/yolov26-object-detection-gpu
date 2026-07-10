#include "Utils.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <getopt.h>
#include <stdexcept>

namespace Utils {

// ============================================================
//  Letterbox
// ============================================================
LetterboxResult letterbox(const cv::Mat& src, int targetSize, cv::Scalar padColor) {
    LetterboxResult lr;
    lr.targetSize = cv::Size(targetSize, targetSize);

    float scaleW = static_cast<float>(targetSize) / src.cols;
    float scaleH = static_cast<float>(targetSize) / src.rows;
    lr.scale     = std::min(scaleW, scaleH);

    int newW = static_cast<int>(std::round(src.cols * lr.scale));
    int newH = static_cast<int>(std::round(src.rows * lr.scale));

    float padW = (targetSize - newW) / 2.0f;
    float padH = (targetSize - newH) / 2.0f;
    lr.pad = cv::Point2f(padW, padH);

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(newW, newH), 0, 0, cv::INTER_LINEAR);

    int top    = static_cast<int>(std::round(padH - 0.1f));
    int bottom = static_cast<int>(std::round(padH + 0.1f));
    int left   = static_cast<int>(std::round(padW - 0.1f));
    int right  = static_cast<int>(std::round(padW + 0.1f));
    // Correct for rounding
    top    = static_cast<int>(padH);
    bottom = targetSize - newH - top;
    left   = static_cast<int>(padW);
    right  = targetSize - newW - left;

    cv::copyMakeBorder(resized, lr.image, top, bottom, left, right,
                       cv::BORDER_CONSTANT, padColor);

    // Adjust pad to exact pixel offsets
    lr.pad = cv::Point2f(static_cast<float>(left), static_cast<float>(top));
    return lr;
}

// ============================================================
//  Inverse letterbox
// ============================================================
cv::Rect inverseLetterbox(const cv::Rect2f& boxNorm,
                           const LetterboxResult& lb,
                           cv::Size origSize) {
    int ts = lb.targetSize.width;

    // From [0,1] in letterboxed space to pixel coords in letterboxed image
    float x1 = boxNorm.x * ts;
    float y1 = boxNorm.y * ts;
    float x2 = (boxNorm.x + boxNorm.width)  * ts;
    float y2 = (boxNorm.y + boxNorm.height) * ts;

    // Remove padding
    x1 -= lb.pad.x;
    y1 -= lb.pad.y;
    x2 -= lb.pad.x;
    y2 -= lb.pad.y;

    // Remove scale
    x1 /= lb.scale;
    y1 /= lb.scale;
    x2 /= lb.scale;
    y2 /= lb.scale;

    // Clip to original frame
    x1 = std::max(0.f, std::min(x1, static_cast<float>(origSize.width  - 1)));
    y1 = std::max(0.f, std::min(y1, static_cast<float>(origSize.height - 1)));
    x2 = std::max(0.f, std::min(x2, static_cast<float>(origSize.width  - 1)));
    y2 = std::max(0.f, std::min(y2, static_cast<float>(origSize.height - 1)));

    return cv::Rect(static_cast<int>(x1), static_cast<int>(y1),
                    static_cast<int>(x2 - x1), static_cast<int>(y2 - y1));
}

// ============================================================
//  NMS
// ============================================================
std::vector<Detection> applyNMS(std::vector<Detection>& dets,
                                 float iouThreshold,
                                 bool agnostic) {
    if (dets.empty()) return {};

    // Sort by confidence descending
    std::sort(dets.begin(), dets.end(),
              [](const Detection& a, const Detection& b){
                  return a.confidence > b.confidence;
              });

    // OpenCV NMS expects cv::Rect and scores separately
    std::vector<cv::Rect>  boxes;
    std::vector<float>     scores;
    boxes.reserve(dets.size());
    scores.reserve(dets.size());

    for (auto& d : dets) {
        // Convert normalised box → integer pixel box for NMS computation
        int x = static_cast<int>(d.boxNorm.x * 10000);
        int y = static_cast<int>(d.boxNorm.y * 10000);
        int w = static_cast<int>(d.boxNorm.width  * 10000);
        int h = static_cast<int>(d.boxNorm.height * 10000);
        if (agnostic) {
            boxes.push_back(cv::Rect(x, y, w, h));
        } else {
            // Offset boxes by class to perform per-class NMS
            int offset = d.classId * 10001;
            boxes.push_back(cv::Rect(x + offset, y + offset, w, h));
        }
        scores.push_back(d.confidence);
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, 0.f, iouThreshold, indices);

    std::vector<Detection> kept;
    kept.reserve(indices.size());
    for (int idx : indices) {
        kept.push_back(dets[idx]);
    }
    return kept;
}

// ============================================================
//  Colour per class
// ============================================================
cv::Scalar classColor(int classId) {
    static const int palette[][3] = {
        {255,56,56},{255,157,151},{255,112,31},{255,178,29},{207,210,49},
        {72,249,10},{146,204,23},{61,219,134},{26,147,52},{0,212,187},
        {44,153,168},{0,194,255},{52,69,147},{100,115,255},{0,24,236},
        {132,56,255},{82,0,133},{203,56,255},{255,149,200},{255,55,199}
    };
    int idx = classId % 20;
    return cv::Scalar(palette[idx][2], palette[idx][1], palette[idx][0]);
}

// ============================================================
//  Draw detections
// ============================================================
void drawDetections(cv::Mat& frame,
                    const std::vector<Detection>& dets,
                    const std::vector<std::string>& classes) {
    for (const auto& d : dets) {
        cv::Scalar color = classColor(d.classId);
        cv::rectangle(frame, d.boxOrig, color, 2);

        std::string label;
        if (d.classId < static_cast<int>(classes.size()))
            label = classes[d.classId];
        else
            label = "class_" + std::to_string(d.classId);
        label += " " + std::to_string(static_cast<int>(d.confidence * 100)) + "%";

        int baseLine = 0;
        cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        cv::Point tl(d.boxOrig.x, std::max(d.boxOrig.y - 1, ts.height));
        cv::rectangle(frame,
                      cv::Point(tl.x, tl.y - ts.height - baseLine),
                      cv::Point(tl.x + ts.width, tl.y + baseLine),
                      color, cv::FILLED);
        cv::putText(frame, label, tl, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }
}

// ============================================================
//  Overlay
// ============================================================
void drawOverlay(cv::Mat& frame, const OverlayInfo& info) {
    // ── Líneas a mostrar ──────────────────────────────────────
    std::string lineDevice = (info.device.find("cuda") != std::string::npos) ? "GPU" : "CPU";
    std::string lineFps;
    { std::ostringstream s; s << std::fixed << std::setprecision(1)
                               << info.avgFps << " FPS"; lineFps = s.str(); }

    std::vector<std::pair<std::string, cv::Scalar>> lines = {
        { info.modelName,       {100, 220, 255} },   // azul claro
        { lineDevice,           {100, 255, 150} },   // verde
        { lineFps,              {255, 220,  80} },   // amarillo
        { info.macAddress,      {180, 180, 255} },   // lila
    };

    // ── Parámetros de fuente escalados al ancho del frame ───
    double scale  = std::max(0.6, frame.cols / 640.0);
    int    thick  = std::max(1, static_cast<int>(scale * 1.5));
    int    dy     = static_cast<int>(32 * scale);
    int    margin = static_cast<int>(10 * scale);
    int    pad    = static_cast<int>(6  * scale);

    // ── Calcular ancho máximo del fondo ─────────────────────
    int maxW = 0;
    for (auto& [txt, _] : lines) {
        int bl = 0;
        auto sz = cv::getTextSize(txt, cv::FONT_HERSHEY_SIMPLEX, scale, thick, &bl);
        maxW = std::max(maxW, sz.width);
    }

    // ── Dibujar fondo semitransparente ──────────────────────
    int boxH = static_cast<int>(lines.size()) * dy + pad * 2;
    cv::Mat roi = frame(cv::Rect(margin, margin,
                                  std::min(maxW + pad * 2, frame.cols - margin * 2),
                                  std::min(boxH, frame.rows - margin * 2)));
    cv::Mat dark(roi.size(), roi.type(), cv::Scalar(0, 0, 0));
    cv::addWeighted(roi, 0.35, dark, 0.65, 0, roi);

    // ── Dibujar texto ────────────────────────────────────────
    int y = margin + pad + dy - static_cast<int>(4 * scale);
    for (auto& [txt, col] : lines) {
        cv::putText(frame, txt, cv::Point(margin + pad, y),
                    cv::FONT_HERSHEY_SIMPLEX, scale, col, thick, cv::LINE_AA);
        y += dy;
    }
}

// ============================================================
//  Load classes
// ============================================================
std::vector<std::string> loadClasses(const std::string& path, int fallbackCount) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "[WARN] classes file not found: " << path
                  << " – using class_N labels\n";
        std::vector<std::string> v;
        v.reserve(fallbackCount);
        for (int i = 0; i < fallbackCount; ++i)
            v.push_back("class_" + std::to_string(i));
        return v;
    }
    std::vector<std::string> v;
    std::string line;
    while (std::getline(f, line)) {
        // trim
        while (!line.empty() && (line.back() == '\r' || line.back() == ' '))
            line.pop_back();
        if (!line.empty())
            v.push_back(line);
    }
    std::cout << "[INFO] Loaded " << v.size() << " classes from " << path << "\n";
    return v;
}

// ============================================================
//  Statistics
// ============================================================
Stats computeStats(std::vector<double> v) {
    Stats s;
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    s.min_ = v.front();
    s.max_ = v.back();
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    s.mean = sum / v.size();
    double sq = 0;
    for (double x : v) sq += (x - s.mean) * (x - s.mean);
    s.stddev = std::sqrt(sq / v.size());
    s.median = (v.size() % 2 == 0)
        ? (v[v.size()/2 - 1] + v[v.size()/2]) / 2.0
        : v[v.size()/2];
    size_t p95idx = static_cast<size_t>(std::ceil(0.95 * v.size())) - 1;
    s.p95 = v[std::min(p95idx, v.size()-1)];
    return s;
}

// ============================================================
//  Argument parsing
// ============================================================
static void printHelp(const char* prog) {
    std::cout <<
"Usage: " << prog << " [OPTIONS]\n"
"\nRequired:\n"
"  --model PATH          Path to ONNX model\n"
"  --model-name NAME     Human-readable model name\n"
"  --source PATH         Input video file\n"
"\nOptional:\n"
"  --device cpu|cuda|cuda_fp16  (default: cpu)\n"
"  --output PATH         Output video path\n"
"  --classes PATH        Classes file (default: config/classes.txt)\n"
"  --imgsz N             Inference resolution (default: 640)\n"
"  --conf FLOAT          Confidence threshold (default: 0.25)\n"
"  --iou FLOAT           IoU NMS threshold (default: 0.45)\n"
"  --warmup N            Warmup frames (default: 20)\n"
"  --max-frames N        Max frames to process (-1=all, default: -1)\n"
"  --show                Display window\n"
"  --save                Save output video (default: on)\n"
"  --no-save             Disable video saving\n"
"  --benchmark-only      Exclude draw+write from timing\n"
"  --include-video-io    Include read+write in totalMs\n"
"  --agnostic-nms        Class-agnostic NMS\n"
"  --output-format auto|raw|end2end  (default: auto)\n"
"  -h, --help            Show this help\n\n";
}

AppConfig parseArgs(int argc, char** argv) {
    AppConfig cfg;
    if (argc < 2) { printHelp(argv[0]); exit(1); }

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nextArg = [&]() -> std::string {
            if (i+1 >= argc) { std::cerr<<"Missing value for "<<a<<"\n"; exit(1); }
            return argv[++i];
        };
        if      (a == "--model")          cfg.modelPath      = nextArg();
        else if (a == "--model-name")     cfg.modelName      = nextArg();
        else if (a == "--source")         cfg.source         = nextArg();
        else if (a == "--device")         cfg.device         = nextArg();
        else if (a == "--output")         cfg.outputPath     = nextArg();
        else if (a == "--classes")        cfg.classesFile    = nextArg();
        else if (a == "--imgsz")          cfg.imgSize        = std::stoi(nextArg());
        else if (a == "--conf")           cfg.confThreshold  = std::stof(nextArg());
        else if (a == "--iou")            cfg.iouThreshold   = std::stof(nextArg());
        else if (a == "--warmup")         cfg.warmupFrames   = std::stoi(nextArg());
        else if (a == "--max-frames")     cfg.maxFrames      = std::stoi(nextArg());
        else if (a == "--show")           cfg.showWindow     = true;
        else if (a == "--save")           cfg.saveVideo      = true;
        else if (a == "--no-save")        cfg.saveVideo      = false;
        else if (a == "--benchmark-only") cfg.benchmarkOnly  = true;
        else if (a == "--include-video-io") cfg.includeVideoIO = true;
        else if (a == "--agnostic-nms")   cfg.agnosticNMS    = true;
        else if (a == "--output-format")  cfg.outputFormat   = nextArg();
        else if (a == "-h" || a == "--help") { printHelp(argv[0]); exit(0); }
        else { std::cerr << "[WARN] Unknown argument: " << a << "\n"; }
    }

    if (cfg.modelPath.empty()) { std::cerr<<"ERROR: --model required\n"; exit(1); }
    if (cfg.source.empty())    { std::cerr<<"ERROR: --source required\n"; exit(1); }
    if (cfg.modelName.empty()) cfg.modelName = cfg.modelPath;

    if (cfg.device != "cpu" && cfg.device != "cuda" && cfg.device != "cuda_fp16") {
        std::cerr << "ERROR: --device must be cpu|cuda|cuda_fp16\n"; exit(1);
    }
    if (cfg.outputFormat != "auto" && cfg.outputFormat != "raw" &&
        cfg.outputFormat != "end2end") {
        std::cerr << "ERROR: --output-format must be auto|raw|end2end\n"; exit(1);
    }

    return cfg;
}

} // namespace Utils
