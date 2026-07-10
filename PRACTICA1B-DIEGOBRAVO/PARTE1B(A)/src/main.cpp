#include "Utils.hpp"
#include "YoloDetector.hpp"
#include "Benchmark.hpp"
#include "SystemMonitor.hpp"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <csignal>
#include <atomic>

namespace fs = std::filesystem;

// ============================================================
//  Logger: stdout + file
// ============================================================
class Logger {
public:
    Logger() = default;
    void open(const std::string& path) {
        fs::create_directories(fs::path(path).parent_path());
        file_.open(path, std::ios::app);
    }
    void log(const std::string& msg) {
        auto t = std::time(nullptr);
        char ts[24];
        std::strftime(ts, sizeof(ts), "[%H:%M:%S]", std::localtime(&t));
        std::string line = std::string(ts) + " " + msg;
        std::cout << line << "\n";
        if (file_.is_open()) file_ << line << "\n" << std::flush;
    }
private:
    std::ofstream file_;
};

static Logger gLog;
static std::atomic<bool> gInterrupt{false};
static void sigHandler(int) { gInterrupt = true; }

// ============================================================
//  Timestamp string
// ============================================================
static std::string timestamp() {
    auto t  = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
    return buf;
}

// ============================================================
//  Ensure directories exist
// ============================================================
static void ensureDirs(const AppConfig& cfg) {
    fs::create_directories("results");
    fs::create_directories("logs");
    if (!cfg.outputPath.empty())
        fs::create_directories(fs::path(cfg.outputPath).parent_path());
}

// ============================================================
//  Print hardware info
// ============================================================
static void printHardwareInfo(const SystemMonitor& mon) {
    const auto& h = mon.hardwareInfo();
    gLog.log("=== Hardware ===");
    gLog.log("  CPU      : " + h.cpuName + " (" + std::to_string(h.cpuCores) + " cores)");
    gLog.log("  RAM      : " + [&]{ std::ostringstream ss;
        ss<<std::fixed<<std::setprecision(0)<<h.totalRamMb<<" MB"; return ss.str();}());
    if (!h.gpus.empty()) {
        auto& g = h.gpus[0];
        gLog.log("  GPU      : " + g.name);
        gLog.log("  VRAM     : " + [&]{ std::ostringstream ss;
            ss<<std::fixed<<std::setprecision(0)<<g.totalVramMb<<" MB"; return ss.str();}());
        gLog.log("  Driver   : " + g.driverVersion);
        gLog.log("  CUDA     : " + g.cudaVersion);
        gLog.log("  Temp     : " + [&]{ std::ostringstream ss;
            ss<<std::fixed<<std::setprecision(0)<<g.temperatureC<<"°C"; return ss.str();}());
    }
    gLog.log("  OpenCV   : " + h.opencvVersion +
             (h.cudaInOpenCV ? " (CUDA enabled)" : " (no CUDA)"));
    gLog.log("  MAC      : " + h.macAddress);
}

// ============================================================
//  Main
// ============================================================
int main(int argc, char** argv) {
    std::signal(SIGINT, sigHandler);

    AppConfig cfg = Utils::parseArgs(argc, argv);

    // Open log
    gLog.open("logs/yolo_benchmark.log");
    gLog.log("=== yolo_benchmark started ===");
    gLog.log("  Model  : " + cfg.modelName + " (" + cfg.modelPath + ")");
    gLog.log("  Source : " + cfg.source);
    gLog.log("  Device : " + cfg.device);
    gLog.log("  Output : " + cfg.outputPath);

    ensureDirs(cfg);

    // Hardware monitor
    SystemMonitor sysmon;
    sysmon.init();
    sysmon.start(1000);
    printHardwareInfo(sysmon);

    // Load classes
    std::vector<std::string> classes = Utils::loadClasses(cfg.classesFile);

    // Open video
    cv::VideoCapture cap(cfg.source);
    if (!cap.isOpened()) {
        gLog.log("[ERROR] Cannot open video: " + cfg.source);
        return 1;
    }

    int origW = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int origH = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double vidFps = cap.get(cv::CAP_PROP_FPS);
    int totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

    gLog.log("  Video  : " + std::to_string(origW) + "x" + std::to_string(origH) +
             " @ " + [&]{ std::ostringstream ss; ss<<std::fixed<<std::setprecision(2)
                <<vidFps; return ss.str();}() + " FPS, " + std::to_string(totalFrames) + " frames");

    // Setup video writer
    cv::VideoWriter writer;
    if (cfg.saveVideo && !cfg.outputPath.empty()) {
        int codec = cv::VideoWriter::fourcc('m','p','4','v');
        writer.open(cfg.outputPath, codec, vidFps, cv::Size(origW, origH));
        if (!writer.isOpened())
            gLog.log("[WARN] Cannot open video writer: " + cfg.outputPath);
        else
            gLog.log("[INFO] Writing video to: " + cfg.outputPath);
    }

    // Create detector
    std::unique_ptr<YoloDetector> detector;
    try {
        detector = std::make_unique<YoloDetector>(cfg, classes);
    } catch (const std::exception& e) {
        gLog.log("[ERROR] " + std::string(e.what()));
        sysmon.stop();
        return 1;
    }

    // Warmup
    gLog.log("[INFO] Running " + std::to_string(cfg.warmupFrames) + " warmup frames...");
    detector->warmup(std::max(1, cfg.warmupFrames));

    // Main benchmark loop
    BenchmarkResult result;
    result.timestamp      = timestamp();
    result.modelName      = cfg.modelName;
    result.modelPath      = cfg.modelPath;
    result.device         = cfg.device;
    result.precision      = (cfg.device == "cuda_fp16") ? "fp16" : "fp32";
    result.videoPath      = cfg.source;
    result.inputWidth     = cfg.imgSize;
    result.inputHeight    = cfg.imgSize;
    result.originalWidth  = origW;
    result.originalHeight = origH;
    result.warmupFrames   = cfg.warmupFrames;
    result.includeVideoIO = cfg.includeVideoIO;
    {
        const auto& h = sysmon.hardwareInfo();
        result.cpuName       = h.cpuName;
        result.macAddress    = h.macAddress;
        result.opencvVersion = h.opencvVersion;
        result.cudaAvailable = h.cudaInOpenCV;
        result.gpuName       = h.gpus.empty() ? "" : h.gpus[0].name;
    }

    // Skip warmup frames from video
    {
        cv::Mat dummy;
        int skipped = 0;
        while (skipped < cfg.warmupFrames) {
            if (!cap.read(dummy) || dummy.empty()) break;
            ++skipped;
        }
        gLog.log("[INFO] Skipped " + std::to_string(skipped) + " warmup frames from video");
        if (skipped < cfg.warmupFrames) {
            gLog.log("[WARN] Video shorter than warmup frames – rewinding");
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
        }
    }

    gLog.log("[INFO] Starting benchmark loop...");

    // Create window before the loop so it appears immediately with focus
    std::string winName = "YOLO Benchmark | " + cfg.modelName + " [" + cfg.device + "]";
    if (cfg.showWindow) {
        cv::namedWindow(winName, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
        // 80 % del tamaño original, sin salirse de pantallas 1080p
        int winH = std::min(static_cast<int>(origH * 0.80), 900);
        int winW = std::min(static_cast<int>(origW * 0.80), 1400);
        cv::resizeWindow(winName, winW, winH);
        cv::moveWindow(winName, 100, 50);   // position top-left so it's visible
        // Show a black frame immediately so the window appears before inference starts
        cv::Mat splash = cv::Mat::zeros(origH, origW, CV_8UC3);
        cv::putText(splash, "Cargando " + cfg.modelName + " [" + cfg.device + "]...",
                    cv::Point(20, origH / 2), cv::FONT_HERSHEY_SIMPLEX,
                    1.0, cv::Scalar(100, 255, 100), 2, cv::LINE_AA);
        cv::imshow(winName, splash);
        cv::waitKey(500);  // 500 ms para que el WM de KDE tenga tiempo de mostrarla
    }

    double wallStart = Utils::nowMs();
    int frameIdx = 0;
    double sumFps = 0.0;
    std::vector<double> recentFps;

    while (!gInterrupt) {
        if (cfg.maxFrames > 0 && frameIdx >= cfg.maxFrames) break;

        FrameTimings timing;

        // --- Read frame ---
        double t0 = Utils::nowMs();
        cv::Mat frame;
        bool ok = cap.read(frame);
        timing.readMs = Utils::nowMs() - t0;

        if (!ok || frame.empty()) break;

        // --- Detect (preprocess + infer + postprocess) ---
        std::vector<Detection> dets;
        try {
            dets = detector->detect(frame, timing);
        } catch (const std::exception& e) {
            gLog.log("[ERROR] Detection failed on frame " + std::to_string(frameIdx)
                     + ": " + e.what());
            break;
        }

        timing.detectionCount = static_cast<int>(dets.size());

        // --- Draw (if not benchmark-only or if saving video) ---
        if (!cfg.benchmarkOnly || cfg.saveVideo || cfg.showWindow) {
            double t1 = Utils::nowMs();

            // Collect overlay info
            sumFps += (timing.inferenceMs + timing.postprocessMs > 0)
                    ? 1000.0 / (timing.inferenceMs + timing.postprocessMs) : 0;

            // Always draw detections and overlay (even in benchmark-only when saving)
            Utils::drawDetections(frame, dets, classes);

            {
                GpuStats gpu = sysmon.getCurrentGpuStats(0);
                Utils::OverlayInfo oi;
                oi.modelName    = cfg.modelName;
                oi.device       = cfg.device;
                oi.precision    = result.precision;
                oi.instantFps   = recentFps.size() >= 2 ? recentFps[recentFps.size()-2] : 0.0;
                oi.avgFps       = frameIdx > 0 ? sumFps / (frameIdx+1) : 0.0;
                oi.inferenceMs  = timing.inferenceMs;
                oi.totalMs      = timing.totalMs;
                oi.detections   = timing.detectionCount;
                oi.ramMb        = sysmon.getProcessRamMb();
                oi.vramMb       = gpu.usedVramMb;
                oi.gpuUsage     = gpu.gpuUsagePct;
                oi.inputSize    = cv::Size(cfg.imgSize, cfg.imgSize);
                oi.macAddress   = result.macAddress;
                Utils::drawOverlay(frame, oi);
            }
            timing.drawMs = Utils::nowMs() - t1;
        }

        // --- Write video ---
        if (cfg.saveVideo && writer.isOpened()) {
            double t2 = Utils::nowMs();
            writer.write(frame);
            timing.writeMs = Utils::nowMs() - t2;
        }

        // --- Show window ---
        if (cfg.showWindow) {
            cv::imshow(winName, frame);
            int key = cv::waitKey(1);
            if (key == 'q' || key == 27) {
                gLog.log("[INFO] Window closed by user");
                break;
            }
        }

        // --- Hardware stats ---
        {
            GpuStats gpu    = sysmon.getCurrentGpuStats(0);
            timing.ramMb    = sysmon.getProcessRamMb();
            timing.vramMb   = gpu.usedVramMb;
            timing.gpuUsage = gpu.gpuUsagePct;
        }

        // --- Compute total ---
        if (cfg.includeVideoIO)
            timing.totalMs = timing.readMs + timing.preprocessMs +
                             timing.inferenceMs + timing.postprocessMs +
                             timing.writeMs;
        else
            timing.totalMs = timing.preprocessMs + timing.inferenceMs +
                             timing.postprocessMs;

        if (timing.totalMs > 0)
            timing.fps = 1000.0 / timing.totalMs;

        recentFps.push_back(timing.fps);
        if (recentFps.size() > 30) recentFps.erase(recentFps.begin());

        result.frames.push_back(timing);
        ++frameIdx;

        // Progress print every 100 frames
        if (frameIdx % 100 == 0) {
            double elapsed = Utils::nowMs() - wallStart;
            gLog.log("[INFO] Frame " + std::to_string(frameIdx) +
                     "  infer=" + Utils::fmtMs(timing.inferenceMs) +
                     "  total=" + Utils::fmtMs(timing.totalMs) +
                     "  fps=" + [&]{ std::ostringstream ss;
                         ss<<std::fixed<<std::setprecision(1)<<timing.fps; return ss.str();}() +
                     "  wall=" + [&]{ std::ostringstream ss;
                         ss<<std::fixed<<std::setprecision(0)<<elapsed/1000.0<<"s"; return ss.str();}());
        }
    }

    double wallEnd = Utils::nowMs();

    if (cfg.showWindow) {
        cv::waitKey(1000);          // pausa 1 s para que se vea el último frame
        cv::destroyWindow(winName);
        cv::waitKey(1);
    }
    writer.release();
    cap.release();
    sysmon.stop();

    if (result.frames.empty()) {
        gLog.log("[ERROR] No frames processed.");
        return 1;
    }

    // Compute stats
    result.stats = Benchmark::computeStats(result.frames);
    result.stats.throughput = result.frames.size() / ((wallEnd - wallStart) / 1000.0);

    Benchmark::printSummary(result);

    // Export results
    Benchmark::exportCSV(result, "results/yolo_benchmark.csv");
    Benchmark::appendJSON(result, "results/yolo_benchmark.json");
    Benchmark::generateComparisonTable("results/yolo_benchmark.csv",
                                        "results/comparison_table.md");

    gLog.log("=== Benchmark complete ===");
    gLog.log("  Processed : " + std::to_string(result.frames.size()) + " frames");
    gLog.log("  Avg FPS   : " + [&]{ std::ostringstream ss;
        ss<<std::fixed<<std::setprecision(2)<<result.stats.avgFps; return ss.str();}());
    gLog.log("  Throughput: " + [&]{ std::ostringstream ss;
        ss<<std::fixed<<std::setprecision(2)<<result.stats.throughput; return ss.str();}() + " fps (wall)");

    return 0;
}
