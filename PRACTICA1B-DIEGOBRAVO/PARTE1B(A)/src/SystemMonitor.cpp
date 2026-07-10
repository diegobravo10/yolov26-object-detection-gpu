#include "SystemMonitor.hpp"
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/dnn.hpp>
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <chrono>
#include <thread>
#include <algorithm>

// ============================================================
SystemMonitor::SystemMonitor() {}

SystemMonitor::~SystemMonitor() { stop(); }

// ============================================================
void SystemMonitor::init() {
    hwInfo_ = collectStaticInfo();
}

// ============================================================
void SystemMonitor::start(int pollIntervalMs) {
    pollIntervalMs_ = pollIntervalMs;
    running_ = true;
    pollThread_ = std::thread(&SystemMonitor::pollLoop, this);
}

void SystemMonitor::stop() {
    running_ = false;
    if (pollThread_.joinable()) pollThread_.join();
}

// ============================================================
void SystemMonitor::pollLoop() {
    while (running_) {
        GpuStats g = queryNvidiaSmi(0);
        double   r = readProcessRamMb();

        {
            std::lock_guard<std::mutex> lk(mtx_);
            liveGpu_  = g;
            liveRamMb_ = r;
            if (r > peakRamMb_) peakRamMb_ = r;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(pollIntervalMs_));
    }
}

// ============================================================
GpuStats SystemMonitor::getCurrentGpuStats(int /*gpuId*/) const {
    std::lock_guard<std::mutex> lk(mtx_);
    return liveGpu_;
}

double SystemMonitor::getProcessRamMb() const {
    std::lock_guard<std::mutex> lk(mtx_);
    return liveRamMb_;
}

double SystemMonitor::getPeakRamMb() const {
    std::lock_guard<std::mutex> lk(mtx_);
    return peakRamMb_;
}

// ============================================================
//  Static hardware info
// ============================================================
HardwareInfo SystemMonitor::collectStaticInfo() {
    HardwareInfo h;
    h.cpuName      = readCpuName();
    h.cpuCores     = readCpuCores();
    h.totalRamMb   = readTotalRamMb();
    h.macAddress   = readMacAddress();
    h.opencvVersion= cv::getVersionString();
    h.cudaInOpenCV = (cv::cuda::getCudaEnabledDeviceCount() > 0);

    // First GPU static info
    GpuStats g = queryNvidiaSmi(0);
    if (!g.name.empty()) h.gpus.push_back(g);

    return h;
}

// ============================================================
//  nvidia-smi queries
// ============================================================
std::string SystemMonitor::readNvidiaSmiField(const std::string& query,
                                               const std::string& format) {
    std::string cmd = "nvidia-smi --query-gpu=" + query +
                      " --format=" + format + " 2>/dev/null";
    FILE* fp = popen(cmd.c_str(), "r");
    if (!fp) return "";
    char buf[512] = {0};
    if (fgets(buf, sizeof(buf), fp)) {
        std::string s(buf);
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' '))
            s.pop_back();
        pclose(fp);
        return s;
    }
    pclose(fp);
    return "";
}

GpuStats SystemMonitor::queryNvidiaSmi(int gpuId) {
    GpuStats g;
    std::string cmd =
        "nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,"
        "temperature.gpu,driver_version,compute_cap "
        "--format=csv,noheader,nounits "
        "--id=" + std::to_string(gpuId) + " 2>/dev/null";

    FILE* fp = popen(cmd.c_str(), "r");
    if (!fp) return g;

    char buf[1024] = {0};
    if (!fgets(buf, sizeof(buf), fp)) { pclose(fp); return g; }
    pclose(fp);

    // Parse CSV: name, mem_total, mem_used, util, temp, driver, compute_cap
    std::istringstream ss(buf);
    std::string tok;
    std::vector<std::string> tokens;
    while (std::getline(ss, tok, ',')) {
        while (!tok.empty() && tok.front() == ' ') tok.erase(tok.begin());
        while (!tok.empty() && (tok.back() == ' ' || tok.back() == '\n' || tok.back() == '\r'))
            tok.pop_back();
        tokens.push_back(tok);
    }

    if (tokens.size() >= 1) g.name         = tokens[0];
    if (tokens.size() >= 2) {
        try { g.totalVramMb = std::stod(tokens[1]); } catch(...) {}
    }
    if (tokens.size() >= 3) {
        try { g.usedVramMb = std::stod(tokens[2]); } catch(...) {}
    }
    if (tokens.size() >= 4) {
        try { g.gpuUsagePct = std::stod(tokens[3]); } catch(...) {}
    }
    if (tokens.size() >= 5) {
        try { g.temperatureC = std::stod(tokens[4]); } catch(...) {}
    }
    if (tokens.size() >= 6) g.driverVersion = tokens[5];

    // CUDA version from nvcc or nvidia-smi
    {
        FILE* fp2 = popen("nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \\K[0-9.]+'", "r");
        if (fp2) {
            char cbuf[64] = {0};
            if (fgets(cbuf, sizeof(cbuf), fp2)) {
                std::string cv(cbuf);
                while (!cv.empty() && (cv.back()=='\n'||cv.back()=='\r'||cv.back()==' '))
                    cv.pop_back();
                g.cudaVersion = cv;
            }
            pclose(fp2);
        }
    }

    return g;
}

// ============================================================
//  Process RAM: read /proc/self/status VmRSS
// ============================================================
double SystemMonitor::readProcessRamMb() {
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
        if (line.find("VmRSS:") != std::string::npos) {
            std::istringstream ss(line);
            std::string key; long val; std::string unit;
            ss >> key >> val >> unit;
            return val / 1024.0;  // kB → MB
        }
    }
    return 0.0;
}

// ============================================================
//  Total RAM: /proc/meminfo
// ============================================================
double SystemMonitor::readTotalRamMb() {
    std::ifstream f("/proc/meminfo");
    std::string line;
    while (std::getline(f, line)) {
        if (line.find("MemTotal:") != std::string::npos) {
            std::istringstream ss(line);
            std::string k; long v;
            ss >> k >> v;
            return v / 1024.0;
        }
    }
    return 0.0;
}

// ============================================================
//  CPU name
// ============================================================
std::string SystemMonitor::readCpuName() {
    std::ifstream f("/proc/cpuinfo");
    std::string line;
    while (std::getline(f, line)) {
        if (line.find("model name") != std::string::npos) {
            auto pos = line.find(':');
            if (pos != std::string::npos) {
                std::string name = line.substr(pos + 2);
                while (!name.empty() && name.front() == ' ') name.erase(name.begin());
                return name;
            }
        }
    }
    return "Unknown CPU";
}

// ============================================================
//  CPU core count
// ============================================================
int SystemMonitor::readCpuCores() {
    int count = 0;
    std::ifstream f("/proc/cpuinfo");
    std::string line;
    while (std::getline(f, line)) {
        if (line.find("processor") != std::string::npos) ++count;
    }
    return count > 0 ? count : 1;
}

// ============================================================
//  MAC address: pick first non-loopback interface
// ============================================================
std::string SystemMonitor::readMacAddress() {
    const std::string netPath = "/sys/class/net/";
    DIR* dir = opendir(netPath.c_str());
    if (!dir) return "00:00:00:00:00:00";

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string iface(entry->d_name);
        if (iface == "." || iface == ".." || iface == "lo") continue;
        std::ifstream addrFile(netPath + iface + "/address");
        if (addrFile.is_open()) {
            std::string mac;
            std::getline(addrFile, mac);
            while (!mac.empty() && (mac.back()=='\n'||mac.back()=='\r'))
                mac.pop_back();
            if (!mac.empty() && mac != "00:00:00:00:00:00") {
                closedir(dir);
                return mac;
            }
        }
    }
    closedir(dir);
    return "00:00:00:00:00:00";
}
