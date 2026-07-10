#pragma once
#include "Utils.hpp"
#include <string>
#include <vector>
#include <map>

// ============================================================
//  Benchmark: accumulate per-frame timings & export results
// ============================================================

struct BenchmarkStats {
    // Per-stage stats
    Utils::Stats read, preprocess, inference, postprocess, draw, write, total;
    // FPS
    double avgFps       = 0.0;
    double throughput   = 0.0;   // frames / s over total wall time
    int    framesProcessed = 0;
    // Hardware
    double avgRamMb     = 0.0;
    double peakRamMb    = 0.0;
    double avgVramMb    = 0.0;
    double peakVramMb   = 0.0;
    double avgGpuUsage  = 0.0;
};

struct BenchmarkResult {
    // Run metadata
    std::string timestamp;
    std::string modelName;
    std::string modelPath;
    std::string device;
    std::string precision;     // fp32 | fp16
    std::string videoPath;
    int    inputWidth      = 0;
    int    inputHeight     = 0;
    int    originalWidth   = 0;
    int    originalHeight  = 0;
    int    warmupFrames    = 0;
    bool   includeVideoIO  = false;
    // Hardware
    std::string gpuName;
    std::string cpuName;
    std::string macAddress;
    std::string opencvVersion;
    bool   cudaAvailable   = false;
    // Per-frame data (post-warmup)
    std::vector<FrameTimings> frames;
    // Computed
    BenchmarkStats stats;
};

class Benchmark {
public:
    // Compute stats from accumulated frames
    static BenchmarkStats computeStats(const std::vector<FrameTimings>& frames);

    // Export single-row CSV (appends if file exists, writes header if new)
    static void exportCSV(const BenchmarkResult& r, const std::string& csvPath);

    // Export JSON (array of runs, overwrites each time)
    static void exportJSON(const std::vector<BenchmarkResult>& results,
                           const std::string& jsonPath);

    // Append JSON entry to existing JSON file (reads + merges)
    static void appendJSON(const BenchmarkResult& r, const std::string& jsonPath);

    // Generate markdown comparison table from CSV
    static void generateComparisonTable(const std::string& csvPath,
                                         const std::string& mdPath);

    // Print a brief summary to stdout
    static void printSummary(const BenchmarkResult& r);
};
