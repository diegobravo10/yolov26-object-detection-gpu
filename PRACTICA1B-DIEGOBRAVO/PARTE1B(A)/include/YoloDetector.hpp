#pragma once
#include "Utils.hpp"
#include "OutputParser.hpp"
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>
#include <memory>

// ============================================================
//  YoloDetector: full ONNX inference pipeline via OpenCV DNN
// ============================================================

class YoloDetector {
public:
    explicit YoloDetector(const AppConfig& cfg,
                          const std::vector<std::string>& classes);

    // Run full pipeline on one frame.
    // Fills timing fields; returns detections in original frame coords.
    std::vector<Detection> detect(const cv::Mat& frame, FrameTimings& timings);

    // Run warm-up passes (no timing recorded)
    void warmup(int passes = 1);

    const AppConfig&                  config()  const { return cfg_; }
    const std::vector<std::string>&   classes() const { return classes_; }
    std::string                        deviceStr() const;
    bool                               isOpened() const;

    // Prints model output layer names and shapes (call after construction)
    void printModelInfo();

private:
    AppConfig                   cfg_;
    std::vector<std::string>    classes_;
    cv::dnn::Net                net_;
    std::vector<std::string>    outLayerNames_;
    std::unique_ptr<OutputParser> parser_;
    bool                        parserInspected_{false};

    void configureBackend();

    // Returns blob and letterbox metadata
    cv::Mat preprocess(const cv::Mat& frame, LetterboxResult& lb) const;

    // Runs net.forward() – only this part is counted as inference
    std::vector<cv::Mat> infer(const cv::Mat& blob, double& inferMs);

    // Parse + NMS + inverse letterbox
    std::vector<Detection> postprocess(const std::vector<cv::Mat>& outputs,
                                        const LetterboxResult& lb,
                                        cv::Size origSize) const;
};
