#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <iomanip>

// ============================================================
//  Shared data structures
// ============================================================

struct Detection {
    cv::Rect2f boxNorm;   // normalised [0..1] in letterboxed space
    cv::Rect   boxOrig;   // pixel coords in original frame
    float      confidence{0.f};
    int        classId{0};
    std::string className;
};

struct LetterboxResult {
    cv::Mat   image;      // padded, resized frame (float32 or uint8)
    float     scale{1.f}; // resize factor applied to both axes
    cv::Point2f pad;      // (left_pad, top_pad) in pixels of the resized image
    cv::Size  targetSize;
};

struct FrameTimings {
    double readMs       = 0.0;
    double preprocessMs = 0.0;
    double inferenceMs  = 0.0;
    double postprocessMs= 0.0;
    double drawMs       = 0.0;
    double writeMs      = 0.0;
    double totalMs      = 0.0;   // depends on --include-video-io
    int    detectionCount = 0;
    double fps          = 0.0;
    double ramMb        = 0.0;
    double vramMb       = 0.0;
    double gpuUsage     = 0.0;
};

struct AppConfig {
    // Model
    std::string modelPath;
    std::string modelName;
    // Source
    std::string source;
    // Inference device: cpu | cuda | cuda_fp16
    std::string device      = "cpu";
    // Output video path
    std::string outputPath;
    // Classes file
    std::string classesFile = "config/classes.txt";
    // Inference parameters
    int    imgSize          = 640;
    float  confThreshold    = 0.25f;
    float  iouThreshold     = 0.45f;
    int    warmupFrames     = 20;
    int    maxFrames        = -1;   // -1 = unlimited
    // Behaviour flags
    bool   showWindow       = false;
    bool   saveVideo        = true;
    bool   benchmarkOnly    = false;  // skip draw+write in timing
    bool   includeVideoIO   = false;  // include read+write in totalMs
    bool   agnosticNMS      = false;
    // Output parser override: auto | raw | end2end
    std::string outputFormat = "auto";
};

// ============================================================
//  Utilities
// ============================================================
namespace Utils {

inline double nowMs() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

// Letterbox: resize keeping AR, pad to square
LetterboxResult letterbox(const cv::Mat& src, int targetSize,
                          cv::Scalar padColor = cv::Scalar(114, 114, 114));

// Inverse-letterbox: map box from letterboxed to original coords
cv::Rect inverseLetterbox(const cv::Rect2f& boxNorm,
                           const LetterboxResult& lb,
                           cv::Size origSize);

// Non-Maximum Suppression
std::vector<Detection> applyNMS(std::vector<Detection>& dets,
                                 float iouThreshold,
                                 bool agnostic = false);

// Reproducible colour per class ID
cv::Scalar classColor(int classId);

// Draw detections + optional overlay text
void drawDetections(cv::Mat& frame,
                    const std::vector<Detection>& dets,
                    const std::vector<std::string>& classes);

struct OverlayInfo {
    std::string modelName;
    std::string device;
    std::string precision;
    double instantFps   = 0.0;
    double avgFps       = 0.0;
    double inferenceMs  = 0.0;
    double totalMs      = 0.0;
    int    detections   = 0;
    double ramMb        = 0.0;
    double vramMb       = 0.0;
    double gpuUsage     = 0.0;
    cv::Size inputSize;
    std::string macAddress;   // shown as evidence
};

void drawOverlay(cv::Mat& frame, const OverlayInfo& info);

// Load class names from file; returns ["class_0","class_1",...] if not found
std::vector<std::string> loadClasses(const std::string& path, int fallbackCount = 80);

// Parse CLI arguments
AppConfig parseArgs(int argc, char** argv);

// Format ms
inline std::string fmtMs(double ms) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2) << ms << " ms";
    return ss.str();
}

// Compute statistics over a vector
struct Stats {
    double mean  = 0, min_ = 0, max_ = 0,
           stddev = 0, median = 0, p95 = 0;
};
Stats computeStats(std::vector<double> v);

} // namespace Utils
