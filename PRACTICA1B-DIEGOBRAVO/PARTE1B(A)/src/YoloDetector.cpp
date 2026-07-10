#include "YoloDetector.hpp"
#include <iostream>
#include <stdexcept>

// ============================================================
YoloDetector::YoloDetector(const AppConfig& cfg,
                             const std::vector<std::string>& classes)
    : cfg_(cfg), classes_(classes)
{
    std::cout << "[INFO] Loading model: " << cfg_.modelPath << "\n";
    net_ = cv::dnn::readNetFromONNX(cfg_.modelPath);
    if (net_.empty()) {
        throw std::runtime_error("Failed to load model: " + cfg_.modelPath);
    }

    configureBackend();

    outLayerNames_ = net_.getUnconnectedOutLayersNames();
    std::cout << "[INFO] Output layers (" << outLayerNames_.size() << "):\n";
    for (auto& n : outLayerNames_)
        std::cout << "       " << n << "\n";

    // Determine output format
    OutputParser::Format fmt = OutputParser::Format::AUTO;
    if      (cfg_.outputFormat == "raw")     fmt = OutputParser::Format::RAW;
    else if (cfg_.outputFormat == "end2end") fmt = OutputParser::Format::END2END;

    parser_ = std::make_unique<OutputParser>(fmt,
                                              static_cast<int>(classes_.size()));
}

// ============================================================
void YoloDetector::configureBackend() {
    if (cfg_.device == "cpu") {
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        std::cout << "[INFO] Backend: OpenCV CPU\n";
        return;
    }

    // Check that OpenCV was compiled with CUDA
    if (cv::cuda::getCudaEnabledDeviceCount() < 1) {
        std::cerr << "\n[ERROR] CUDA device requested but OpenCV reports no CUDA-capable GPU.\n";
        std::cerr << "  Diagnosis steps:\n";
        std::cerr << "    1. Verify GPU: nvidia-smi\n";
        std::cerr << "    2. Check OpenCV CUDA support: python3 -c \"import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())\"\n";
        std::cerr << "    3. Rebuild OpenCV with -DWITH_CUDA=ON -DOPENCV_DNN_CUDA=ON\n";
        std::cerr << "    4. Check CUDA toolkit: nvcc --version\n";
        std::cerr << "  NOT falling back to CPU – exiting to preserve fair comparison.\n\n";
        throw std::runtime_error("CUDA not available in OpenCV");
    }

    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net_.enableFusion(false); // workaround: fuseLayers assertion fails on some ONNX models
    if (cfg_.device == "cuda_fp16") {
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
        std::cout << "[INFO] Backend: CUDA FP16\n";
    } else {
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        std::cout << "[INFO] Backend: CUDA FP32\n";
    }
}

// ============================================================
bool YoloDetector::isOpened() const { return !net_.empty(); }

// ============================================================
std::string YoloDetector::deviceStr() const {
    if (cfg_.device == "cuda_fp16") return "CUDA FP16";
    if (cfg_.device == "cuda")      return "CUDA FP32";
    return "CPU";
}

// ============================================================
void YoloDetector::printModelInfo() {
    // Run a dummy forward to get output shapes
    cv::Mat dummy = cv::Mat::zeros(cfg_.imgSize, cfg_.imgSize, CV_8UC3);
    cv::Mat blob = cv::dnn::blobFromImage(dummy, 1.0 / 255.0,
                                           cv::Size(cfg_.imgSize, cfg_.imgSize),
                                           cv::Scalar(), true, false);
    net_.setInput(blob);
    std::vector<cv::Mat> outputs;
    net_.forward(outputs, outLayerNames_);

    bool ok = parser_->inspect(outputs, outLayerNames_);
    if (!ok) throw std::runtime_error("Ambiguous output format – use --output-format");
    parserInspected_ = true;
}

// ============================================================
void YoloDetector::warmup(int passes) {
    cv::Mat dummy = cv::Mat::zeros(cfg_.imgSize, cfg_.imgSize, CV_8UC3);
    LetterboxResult lb;
    cv::Mat blob = preprocess(dummy, lb);
    for (int i = 0; i < passes; ++i) {
        double dummy_ms = 0;
        infer(blob, dummy_ms);
    }
}

// ============================================================
cv::Mat YoloDetector::preprocess(const cv::Mat& frame, LetterboxResult& lb) const {
    lb = Utils::letterbox(frame, cfg_.imgSize);
    cv::Mat blob = cv::dnn::blobFromImage(
        lb.image,
        1.0 / 255.0,
        cv::Size(cfg_.imgSize, cfg_.imgSize),
        cv::Scalar(),   // no mean subtraction
        true,           // swapRB: BGR→RGB
        false           // no crop
    );
    return blob;
}

// ============================================================
std::vector<cv::Mat> YoloDetector::infer(const cv::Mat& blob, double& inferMs) {
    net_.setInput(blob);
    std::vector<cv::Mat> outputs;
    double t0 = Utils::nowMs();
    net_.forward(outputs, outLayerNames_);
    inferMs = Utils::nowMs() - t0;
    return outputs;
}

// ============================================================
std::vector<Detection> YoloDetector::postprocess(
        const std::vector<cv::Mat>& outputs,
        const LetterboxResult& lb,
        cv::Size origSize) const
{
    auto dets = parser_->parse(outputs,
                                cfg_.confThreshold,
                                cfg_.iouThreshold,
                                cfg_.agnosticNMS);

    const float ts = static_cast<float>(cfg_.imgSize);
    const bool isEnd2End = (parser_->resolved() == OutputParser::Resolved::END2END_6);

    for (auto& d : dets) {
        cv::Rect2f normBox = d.boxNorm;

        if (isEnd2End) {
            // parseEnd2End6 stores (x1, y1, w, h) in pixel coords → just normalise
            normBox.x      /= ts;
            normBox.y      /= ts;
            normBox.width  /= ts;
            normBox.height /= ts;
        } else {
            // RAW formats: coords may be absolute pixels (cx,cy,w,h) or [0,1]
            bool absCoords = (std::abs(normBox.x) > 2.f || std::abs(normBox.y) > 2.f);
            if (absCoords) {
                float cx = normBox.x, cy = normBox.y;
                float w  = normBox.width, h = normBox.height;
                if (cx - w/2.f < 0.f || cy - h/2.f < 0.f) {
                    // already x1,y1,w,h in pixels
                    normBox.x      /= ts;
                    normBox.y      /= ts;
                    normBox.width  /= ts;
                    normBox.height /= ts;
                } else {
                    // cx,cy,w,h in pixels → convert to x1,y1 normalised
                    normBox.x      = (cx - w/2.f) / ts;
                    normBox.y      = (cy - h/2.f) / ts;
                    normBox.width  = w / ts;
                    normBox.height = h / ts;
                }
            } else {
                // [0,1] range — check if centre format
                float cx = normBox.x, cy = normBox.y;
                float w  = normBox.width, h = normBox.height;
                if (cx > w/2.f && cy > h/2.f && cx < 1.f && cy < 1.f) {
                    normBox.x = cx - w/2.f;
                    normBox.y = cy - h/2.f;
                }
            }
        }

        d.boxOrig = Utils::inverseLetterbox(normBox, lb, origSize);

        if (!classes_.empty() && d.classId < static_cast<int>(classes_.size()))
            d.className = classes_[d.classId];
        else
            d.className = "class_" + std::to_string(d.classId);
    }

    return dets;
}

// ============================================================
std::vector<Detection> YoloDetector::detect(const cv::Mat& frame,
                                              FrameTimings& timings) {
    cv::Size origSize = frame.size();

    // Preprocess
    double t0 = Utils::nowMs();
    LetterboxResult lb;
    cv::Mat blob = preprocess(frame, lb);
    timings.preprocessMs = Utils::nowMs() - t0;

    // First frame: inspect parser
    if (!parserInspected_) {
        double dummyMs = 0;
        auto outputs = infer(blob, dummyMs);
        bool ok = parser_->inspect(outputs, outLayerNames_);
        if (!ok) throw std::runtime_error("Ambiguous output format – use --output-format");
        parserInspected_ = true;
        // Use this inference for the first measurement
        timings.inferenceMs = dummyMs;
        t0 = Utils::nowMs();
        auto dets = postprocess(outputs, lb, origSize);
        timings.postprocessMs = Utils::nowMs() - t0;
        return dets;
    }

    // Inference
    auto outputs = infer(blob, timings.inferenceMs);

    // Postprocess
    t0 = Utils::nowMs();
    auto dets = postprocess(outputs, lb, origSize);
    timings.postprocessMs = Utils::nowMs() - t0;

    return dets;
}
