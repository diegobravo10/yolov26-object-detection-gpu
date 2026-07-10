#pragma once
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>

// ============================================================
//  SystemMonitor: hardware info + background GPU/RAM polling
// ============================================================

struct GpuStats {
    std::string name;
    double totalVramMb  = 0.0;
    double usedVramMb   = 0.0;
    double gpuUsagePct  = 0.0;
    double temperatureC = 0.0;
    std::string driverVersion;
    std::string cudaVersion;
};

struct HardwareInfo {
    std::string cpuName;
    int         cpuCores       = 0;
    double      totalRamMb     = 0.0;
    bool        cudaInOpenCV   = false;
    std::string opencvVersion;
    std::string macAddress;
    std::vector<GpuStats> gpus;  // one per detected GPU
};

class SystemMonitor {
public:
    SystemMonitor();
    ~SystemMonitor();

    // Collect static info (CPU, RAM total, OpenCV flags, MAC)
    // Must be called before start()
    void init();

    // Start background polling thread (polls ~every pollIntervalMs)
    void start(int pollIntervalMs = 1000);
    void stop();

    // Snapshot of current GPU/RAM state (thread-safe)
    GpuStats  getCurrentGpuStats(int gpuId = 0) const;
    double    getProcessRamMb()  const;
    double    getPeakRamMb()     const;

    const HardwareInfo& hardwareInfo() const { return hwInfo_; }

private:
    HardwareInfo    hwInfo_;
    mutable std::mutex mtx_;

    // Live values updated by poll thread
    GpuStats        liveGpu_;
    double          liveRamMb_{0.0};
    double          peakRamMb_{0.0};

    std::thread     pollThread_;
    std::atomic<bool> running_{false};
    int             pollIntervalMs_{1000};

    void pollLoop();

    // Helpers
    HardwareInfo    collectStaticInfo();
    GpuStats        queryNvidiaSmi(int gpuId = 0);
    double          readProcessRamMb();
    std::string     readMacAddress();
    std::string     readCpuName();
    int             readCpuCores();
    double          readTotalRamMb();
    std::string     readNvidiaSmiField(const std::string& query,
                                        const std::string& format);
};
