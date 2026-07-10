#include "Benchmark.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <ctime>
#include <filesystem>

namespace fs = std::filesystem;

// ============================================================
//  computeStats
// ============================================================
BenchmarkStats Benchmark::computeStats(const std::vector<FrameTimings>& frames) {
    BenchmarkStats s;
    if (frames.empty()) return s;
    s.framesProcessed = static_cast<int>(frames.size());

    auto extract = [&](auto field) {
        std::vector<double> v;
        v.reserve(frames.size());
        for (auto& f : frames) v.push_back(f.*field);
        return v;
    };

    s.read        = Utils::computeStats(extract(&FrameTimings::readMs));
    s.preprocess  = Utils::computeStats(extract(&FrameTimings::preprocessMs));
    s.inference   = Utils::computeStats(extract(&FrameTimings::inferenceMs));
    s.postprocess = Utils::computeStats(extract(&FrameTimings::postprocessMs));
    s.draw        = Utils::computeStats(extract(&FrameTimings::drawMs));
    s.write       = Utils::computeStats(extract(&FrameTimings::writeMs));
    s.total       = Utils::computeStats(extract(&FrameTimings::totalMs));

    if (s.total.mean > 0)
        s.avgFps = 1000.0 / s.total.mean;

    // Hardware averages
    std::vector<double> rams, vrams, gpus;
    for (auto& f : frames) {
        rams.push_back(f.ramMb);
        vrams.push_back(f.vramMb);
        gpus.push_back(f.gpuUsage);
    }
    auto avg = [](const std::vector<double>& v) {
        return v.empty() ? 0.0
             : std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    };
    auto peak = [](const std::vector<double>& v) {
        return v.empty() ? 0.0 : *std::max_element(v.begin(), v.end());
    };

    s.avgRamMb    = avg(rams);
    s.peakRamMb   = peak(rams);
    s.avgVramMb   = avg(vrams);
    s.peakVramMb  = peak(vrams);
    s.avgGpuUsage = avg(gpus);

    return s;
}

// ============================================================
//  Timestamp helper
// ============================================================
static std::string nowTimestamp() {
    std::time_t t = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
    return buf;
}

// ============================================================
//  Export CSV
// ============================================================
void Benchmark::exportCSV(const BenchmarkResult& r, const std::string& csvPath) {
    bool newFile = !fs::exists(csvPath);

    std::ofstream f(csvPath, std::ios::app);
    if (!f.is_open()) {
        std::cerr << "[ERROR] Cannot open CSV: " << csvPath << "\n";
        return;
    }

    if (newFile) {
        f << "timestamp,model_name,model_path,device,precision,video,"
             "input_width,input_height,original_width,original_height,"
             "frames,warmup_frames,"
             "average_fps,min_fps,max_fps,fps_std,"
             "average_read_ms,average_preprocess_ms,average_inference_ms,"
             "average_postprocess_ms,average_draw_ms,average_write_ms,average_total_ms,"
             "p50_inference_ms,p95_inference_ms,p50_total_ms,p95_total_ms,"
             "average_ram_mb,peak_ram_mb,average_vram_mb,peak_vram_mb,average_gpu_usage,"
             "gpu_name,cpu_name,mac_address,opencv_version,cuda_available\n";
    }

    const auto& st = r.stats;
    double minFps  = st.total.max_ > 0 ? 1000.0/st.total.max_ : 0;
    double maxFps  = st.total.min_ > 0 ? 1000.0/st.total.min_ : 0;
    double fpsstd  = 0.0;
    // approximate FPS std from total ms std
    if (st.total.mean > 0)
        fpsstd = (st.total.stddev / (st.total.mean * st.total.mean)) * 1000.0;

    auto q = [](const std::string& s) -> std::string {
        return "\"" + s + "\"";
    };

    f << std::fixed << std::setprecision(4);
    f << q(r.timestamp)     << ","
      << q(r.modelName)     << ","
      << q(r.modelPath)     << ","
      << q(r.device)        << ","
      << q(r.precision)     << ","
      << q(r.videoPath)     << ","
      << r.inputWidth       << ","
      << r.inputHeight      << ","
      << r.originalWidth    << ","
      << r.originalHeight   << ","
      << r.stats.framesProcessed << ","
      << r.warmupFrames     << ","
      << st.avgFps          << ","
      << minFps             << ","
      << maxFps             << ","
      << fpsstd             << ","
      << st.read.mean       << ","
      << st.preprocess.mean << ","
      << st.inference.mean  << ","
      << st.postprocess.mean<< ","
      << st.draw.mean       << ","
      << st.write.mean      << ","
      << st.total.mean      << ","
      << st.inference.median<< ","
      << st.inference.p95   << ","
      << st.total.median    << ","
      << st.total.p95       << ","
      << st.avgRamMb        << ","
      << st.peakRamMb       << ","
      << st.avgVramMb       << ","
      << st.peakVramMb      << ","
      << st.avgGpuUsage     << ","
      << q(r.gpuName)       << ","
      << q(r.cpuName)       << ","
      << q(r.macAddress)    << ","
      << q(r.opencvVersion) << ","
      << (r.cudaAvailable ? "true" : "false")
      << "\n";

    std::cout << "[INFO] CSV appended: " << csvPath << "\n";
}

// ============================================================
//  JSON helpers
// ============================================================
static std::string statsToJson(const Utils::Stats& s, int indent = 8) {
    std::string sp(indent, ' ');
    std::ostringstream o;
    o << std::fixed << std::setprecision(4);
    o << "{\n"
      << sp << "  \"mean\":"   << s.mean   << ",\n"
      << sp << "  \"min\":"    << s.min_   << ",\n"
      << sp << "  \"max\":"    << s.max_   << ",\n"
      << sp << "  \"stddev\":" << s.stddev << ",\n"
      << sp << "  \"median\":" << s.median << ",\n"
      << sp << "  \"p95\":"    << s.p95    << "\n"
      << sp << "}";
    return o.str();
}

static std::string escapeJson(const std::string& s) {
    std::string r;
    for (char c : s) {
        if      (c == '"')  r += "\\\"";
        else if (c == '\\') r += "\\\\";
        else if (c == '\n') r += "\\n";
        else                r += c;
    }
    return r;
}

// ============================================================
//  Export JSON (append to array in file)
// ============================================================
void Benchmark::appendJSON(const BenchmarkResult& r, const std::string& jsonPath) {
    // Read existing JSON or start fresh
    std::vector<std::string> existing;
    std::ifstream fin(jsonPath);
    if (fin.is_open()) {
        // Simple approach: find each JSON object and collect
        std::string content((std::istreambuf_iterator<char>(fin)),
                              std::istreambuf_iterator<char>());
        fin.close();
        // Strip outer [] if present
        auto a = content.find('[');
        auto b = content.rfind(']');
        if (a != std::string::npos && b != std::string::npos && b > a)
            existing.push_back(content.substr(a+1, b-a-1));
    }

    // Build new entry
    const auto& st = r.stats;
    std::ostringstream e;
    e << std::fixed << std::setprecision(4);
    e << "  {\n"
      << "    \"timestamp\": \""         << escapeJson(r.timestamp)    << "\",\n"
      << "    \"model_name\": \""        << escapeJson(r.modelName)    << "\",\n"
      << "    \"model_path\": \""        << escapeJson(r.modelPath)    << "\",\n"
      << "    \"device\": \""            << escapeJson(r.device)       << "\",\n"
      << "    \"precision\": \""         << escapeJson(r.precision)    << "\",\n"
      << "    \"video\": \""             << escapeJson(r.videoPath)    << "\",\n"
      << "    \"input_size\": ["         << r.inputWidth << "," << r.inputHeight << "],\n"
      << "    \"original_size\": ["      << r.originalWidth << "," << r.originalHeight << "],\n"
      << "    \"frames\": "              << st.framesProcessed << ",\n"
      << "    \"warmup_frames\": "       << r.warmupFrames << ",\n"
      << "    \"average_fps\": "         << st.avgFps << ",\n"
      << "    \"statistics\": {\n"
      << "      \"read_ms\": "           << statsToJson(st.read, 6) << ",\n"
      << "      \"preprocess_ms\": "     << statsToJson(st.preprocess, 6) << ",\n"
      << "      \"inference_ms\": "      << statsToJson(st.inference, 6) << ",\n"
      << "      \"postprocess_ms\": "    << statsToJson(st.postprocess, 6) << ",\n"
      << "      \"draw_ms\": "           << statsToJson(st.draw, 6) << ",\n"
      << "      \"write_ms\": "          << statsToJson(st.write, 6) << ",\n"
      << "      \"total_ms\": "          << statsToJson(st.total, 6) << "\n"
      << "    },\n"
      << "    \"hardware\": {\n"
      << "      \"gpu_name\": \""         << escapeJson(r.gpuName)       << "\",\n"
      << "      \"cpu_name\": \""         << escapeJson(r.cpuName)       << "\",\n"
      << "      \"mac_address\": \""      << escapeJson(r.macAddress)    << "\",\n"
      << "      \"opencv_version\": \""   << escapeJson(r.opencvVersion) << "\",\n"
      << "      \"cuda_available\": "     << (r.cudaAvailable ? "true" : "false") << ",\n"
      << "      \"avg_ram_mb\": "         << st.avgRamMb  << ",\n"
      << "      \"peak_ram_mb\": "        << st.peakRamMb << ",\n"
      << "      \"avg_vram_mb\": "        << st.avgVramMb  << ",\n"
      << "      \"peak_vram_mb\": "       << st.peakVramMb << ",\n"
      << "      \"avg_gpu_usage_pct\": "  << st.avgGpuUsage << "\n"
      << "    }\n"
      << "  }";

    // Write combined JSON
    std::ofstream fout(jsonPath);
    fout << "[\n";
    // Write existing entries
    bool hadContent = false;
    for (auto& prev : existing) {
        if (prev.find('{') == std::string::npos) continue;
        if (hadContent) fout << ",\n";
        fout << prev;
        hadContent = true;
    }
    if (hadContent) fout << ",\n";
    fout << e.str() << "\n]\n";

    std::cout << "[INFO] JSON updated: " << jsonPath << "\n";
}

// ============================================================
//  Generate markdown comparison table
// ============================================================
void Benchmark::generateComparisonTable(const std::string& csvPath,
                                         const std::string& mdPath) {
    std::ifstream f(csvPath);
    if (!f.is_open()) {
        std::cerr << "[WARN] Cannot read CSV for comparison table: " << csvPath << "\n";
        return;
    }

    struct Row {
        std::string model, device, precision;
        double avgFps = 0, avgInferMs = 0, avgTotalMs = 0, peakVramMb = 0, peakRamMb = 0;
        double p95InferMs = 0;
        int frames = 0;
    };

    std::vector<Row> rows;
    std::string line;
    bool header = true;
    while (std::getline(f, line)) {
        if (header) { header = false; continue; }
        if (line.empty()) continue;

        // Simple CSV parse (handles quoted fields)
        std::vector<std::string> cols;
        std::istringstream ss(line);
        std::string tok;
        bool inQuote = false;
        std::string cur;
        for (char c : line) {
            if (c == '"') { inQuote = !inQuote; continue; }
            if (c == ',' && !inQuote) { cols.push_back(cur); cur.clear(); }
            else cur += c;
        }
        cols.push_back(cur);

        if (cols.size() < 37) continue;
        Row r;
        r.model       = cols[1];
        r.device      = cols[3];
        r.precision   = cols[4];
        try {
            r.avgFps      = std::stod(cols[12]);
            r.avgInferMs  = std::stod(cols[18]);
            r.avgTotalMs  = std::stod(cols[22]);
            r.p95InferMs  = std::stod(cols[24]);
            r.peakRamMb   = std::stod(cols[29]);
            r.peakVramMb  = std::stod(cols[31]);
            r.frames      = std::stoi(cols[10]);
        } catch (...) {}
        rows.push_back(r);
    }

    if (rows.empty()) { std::cerr << "[WARN] No rows in CSV.\n"; return; }

    std::ofstream md(mdPath);
    md << "# YOLO Benchmark Comparison\n\n";
    md << "Generated: " << nowTimestamp() << "\n\n";
    md << "## Results Table\n\n";
    md << "| Model | Device | Precision | Avg FPS | Avg Infer (ms) | P95 Infer (ms) | Avg Total (ms) | Peak RAM (MB) | Peak VRAM (MB) | Frames |\n";
    md << "|-------|--------|-----------|---------|---------------|---------------|---------------|---------------|----------------|--------|\n";
    for (auto& r : rows) {
        md << std::fixed << std::setprecision(2);
        md << "| " << r.model    << " | " << r.device   << " | " << r.precision
           << " | " << r.avgFps  << " | " << r.avgInferMs << " | " << r.p95InferMs
           << " | " << r.avgTotalMs << " | " << r.peakRamMb
           << " | " << r.peakVramMb << " | " << r.frames << " |\n";
    }

    // Compute speedups
    md << "\n## Speedup Analysis\n\n";
    auto findRow = [&](const std::string& model, const std::string& device) -> const Row* {
        for (auto& r : rows)
            if (r.model == model && (r.device == device || r.device.find(device) != std::string::npos))
                return &r;
        return nullptr;
    };

    // Collect unique models
    std::vector<std::string> models;
    for (auto& r : rows) {
        if (std::find(models.begin(), models.end(), r.model) == models.end())
            models.push_back(r.model);
    }

    for (auto& model : models) {
        const Row* cpuRow  = findRow(model, "cpu");
        const Row* cudaRow = findRow(model, "cuda");
        if (cpuRow && cudaRow && cpuRow->avgTotalMs > 0 && cudaRow->avgTotalMs > 0) {
            double speedup = cpuRow->avgTotalMs / cudaRow->avgTotalMs;
            md << std::fixed << std::setprecision(2);
            md << "- **" << model << "** GPU vs CPU speedup: **" << speedup << "x**"
               << " (" << cpuRow->avgFps << " → " << cudaRow->avgFps << " FPS)\n";
        }
    }

    if (models.size() >= 2) {
        md << "\n### Cross-model comparison\n\n";
        const Row* m0cpu  = findRow(models[0], "cpu");
        const Row* m1cpu  = findRow(models[1], "cpu");
        const Row* m0cuda = findRow(models[0], "cuda");
        const Row* m1cuda = findRow(models[1], "cuda");

        if (m0cpu && m1cpu)
            md << std::fixed << std::setprecision(2)
               << "- CPU FPS diff (" << models[0] << " vs " << models[1] << "): "
               << m0cpu->avgFps - m1cpu->avgFps << " FPS\n";
        if (m0cuda && m1cuda)
            md << "- GPU FPS diff (" << models[0] << " vs " << models[1] << "): "
               << m0cuda->avgFps - m1cuda->avgFps << " FPS\n";

        // Best model analysis
        std::string fastestCpu  = (m0cpu  && m1cpu  && m0cpu->avgFps  > m1cpu->avgFps)
                                  ? models[0] : (models.size()>1 ? models[1] : "");
        std::string fastestGpu  = (m0cuda && m1cuda && m0cuda->avgFps > m1cuda->avgFps)
                                  ? models[0] : (models.size()>1 ? models[1] : "");
        std::string lowestVram  = "";
        {
            std::string best; double bv = 1e18;
            for (auto& r : rows)
                if (r.device.find("cuda") != std::string::npos && r.peakVramMb > 0 && r.peakVramMb < bv)
                    { bv = r.peakVramMb; best = r.model; }
            lowestVram = best;
        }
        std::string lowestLatency = "";
        {
            std::string best; double bv = 1e18;
            for (auto& r : rows)
                if (r.avgInferMs > 0 && r.avgInferMs < bv)
                    { bv = r.avgInferMs; best = r.model + " (" + r.device + ")"; }
            lowestLatency = best;
        }

        md << "\n### Summary\n\n";
        if (!fastestCpu.empty())  md << "- Fastest on CPU: **" << fastestCpu << "**\n";
        if (!fastestGpu.empty())  md << "- Fastest on GPU: **" << fastestGpu << "**\n";
        if (!lowestVram.empty())  md << "- Lowest VRAM:    **" << lowestVram << "**\n";
        if (!lowestLatency.empty()) md << "- Lowest latency: **" << lowestLatency << "**\n";
    }

    std::cout << "[INFO] Comparison table: " << mdPath << "\n";
}

// ============================================================
//  Print summary
// ============================================================
void Benchmark::printSummary(const BenchmarkResult& r) {
    const auto& st = r.stats;
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "  BENCHMARK SUMMARY\n";
    std::cout << "  Model   : " << r.modelName  << "\n";
    std::cout << "  Device  : " << r.device     << " (" << r.precision << ")\n";
    std::cout << "  Frames  : " << st.framesProcessed << " (warmup=" << r.warmupFrames << ")\n";
    std::cout << "----------------------------------------\n";
    auto row = [](const std::string& label, const Utils::Stats& s) {
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  " << std::left << std::setw(14) << label
                  << " avg=" << std::setw(8) << s.mean
                  << " min=" << std::setw(8) << s.min_
                  << " max=" << std::setw(8) << s.max_
                  << " p95=" << std::setw(8) << s.p95
                  << " ms\n";
    };
    row("Preprocess",  st.preprocess);
    row("Inference",   st.inference);
    row("Postprocess", st.postprocess);
    if (st.draw.mean > 0)  row("Draw",      st.draw);
    if (st.write.mean > 0) row("Write",     st.write);
    row("Total",       st.total);
    std::cout << "----------------------------------------\n";
    std::cout << "  Avg FPS : " << std::fixed << std::setprecision(2) << st.avgFps << "\n";
    std::cout << "  Peak RAM: " << st.peakRamMb << " MB\n";
    if (st.peakVramMb > 0)
        std::cout << "  Peak VRAM:" << st.peakVramMb << " MB\n";
    std::cout << "========================================\n\n";
}
