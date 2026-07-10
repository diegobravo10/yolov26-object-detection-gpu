#include "OutputParser.hpp"
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

// ============================================================
OutputParser::OutputParser(Format fmt, int numClasses)
    : format_(fmt), numClasses_(numClasses) {}

// ============================================================
//  Format name helper
// ============================================================
std::string OutputParser::formatName() const {
    switch (resolved_) {
        case Resolved::RAW_CxN:   return "RAW_CxN  [1, 4+C, N]";
        case Resolved::RAW_NxC:   return "RAW_NxC  [1, N, 4+C]";
        case Resolved::OBJ_NxC:   return "OBJ_NxC  [1, N, 5+C]";
        case Resolved::END2END_6: return "END2END  [1, N, 6]";
        default:                  return "UNKNOWN";
    }
}

// ============================================================
//  Inspect: print tensor info, resolve AUTO
// ============================================================
bool OutputParser::inspect(const std::vector<cv::Mat>& outputs,
                            const std::vector<std::string>& outLayerNames) {
    std::cout << "\n=== Model output tensors ===\n";
    for (size_t i = 0; i < outputs.size(); ++i) {
        const cv::Mat& o = outputs[i];
        std::string name = (i < outLayerNames.size()) ? outLayerNames[i] : "output_" + std::to_string(i);
        std::cout << "  [" << i << "] " << name << "  dims=" << o.dims << "  shape=";
        for (int d = 0; d < o.dims; ++d)
            std::cout << o.size[d] << (d < o.dims-1 ? "x" : "");
        std::cout << "  type=" << o.type() << "\n";
    }

    if (format_ == Format::RAW) {
        // User forced raw format; try to determine sub-format
        resolved_ = autoResolve(outputs[0]);
        if (resolved_ == Resolved::END2END_6) {
            // User said raw but shape looks end2end — warn
            std::cerr << "[WARN] --output-format raw but tensor looks END2END.\n";
        }
    } else if (format_ == Format::END2END) {
        resolved_ = Resolved::END2END_6;
    } else {
        // AUTO
        resolved_ = autoResolve(outputs[0]);
        if (resolved_ == Resolved::UNKNOWN) {
            const cv::Mat& o = outputs[0];
            std::cerr << "\n[ERROR] Cannot auto-detect output format.\n"
                      << "        Tensor shape: ";
            for (int d = 0; d < o.dims; ++d)
                std::cerr << o.size[d] << (d < o.dims-1 ? "x" : "");
            std::cerr << "\n"
                      << "        Please re-run with --output-format raw or --output-format end2end\n"
                      << "        and optionally --classes to provide the class count.\n\n";
            return false;
        }
    }

    std::cout << "  Resolved format: " << formatName() << "\n";
    std::cout << "  numClasses used: " << numClasses_ << "\n\n";
    inspected_ = true;
    return true;
}

// ============================================================
//  autoResolve
// ============================================================
OutputParser::Resolved OutputParser::autoResolve(const cv::Mat& out) const {
    if (out.dims != 3) return Resolved::UNKNOWN;

    int A = out.size[1];  // second dimension
    int B = out.size[2];  // third dimension

    // --- End-to-end: exactly 6 values per row ---
    if (B == 6 && A > 1)  return Resolved::END2END_6;
    if (A == 6 && B > 1)  return Resolved::END2END_6; // unusual transposed

    // If we know numClasses, try exact matching first
    if (numClasses_ > 0) {
        if (B == 4 + numClasses_)  return Resolved::RAW_NxC;
        if (B == 5 + numClasses_)  return Resolved::OBJ_NxC;
        if (A == 4 + numClasses_)  return Resolved::RAW_CxN;
        if (A == 5 + numClasses_) {
            // Unusual but possible — treat as transposed OBJ
            std::cerr << "[WARN] Shape [1," << A << "," << B
                      << "] matches 5+C in dimension 1 — treating as OBJ_NxC transposed.\n";
            return Resolved::RAW_CxN;
        }
    }

    // Heuristic based on aspect ratio
    // [1, small, large] → likely [1, C, N] where C is small
    if (A < B) {
        // A could be 4+C
        int inferredC = A - 4;
        if (inferredC > 0) {
            if (numClasses_ == 0 || inferredC == numClasses_)
                return Resolved::RAW_CxN;
        }
        inferredC = A - 5;
        if (inferredC > 0) {
            if (numClasses_ == 0 || inferredC == numClasses_)
                return Resolved::RAW_CxN;  // treat as CxN, objectness is col 4
        }
    }

    // [1, large, small] → likely [1, N, C]
    if (A > B) {
        int inferredC = B - 4;
        if (inferredC > 0) {
            if (numClasses_ == 0 || inferredC == numClasses_)
                return Resolved::RAW_NxC;
        }
        inferredC = B - 5;
        if (inferredC > 0) {
            if (numClasses_ == 0 || inferredC == numClasses_)
                return Resolved::OBJ_NxC;
        }
    }

    return Resolved::UNKNOWN;
}

// ============================================================
//  parse
// ============================================================
std::vector<Detection> OutputParser::parse(const std::vector<cv::Mat>& outputs,
                                            float confThreshold,
                                            float iouThreshold,
                                            bool  agnosticNMS) const {
    if (!inspected_) {
        throw std::runtime_error("OutputParser::inspect() must be called before parse()");
    }
    if (outputs.empty()) return {};

    std::vector<Detection> dets;

    switch (resolved_) {
        case Resolved::RAW_CxN:
            dets = parseRawCxN(outputs[0], confThreshold);
            break;
        case Resolved::RAW_NxC:
            dets = parseRawNxC(outputs[0], confThreshold);
            break;
        case Resolved::OBJ_NxC:
            dets = parseObjNxC(outputs[0], confThreshold);
            break;
        case Resolved::END2END_6:
            dets = parseEnd2End6(outputs[0], confThreshold);
            break;
        default:
            throw std::runtime_error("Cannot parse: format is UNKNOWN");
    }

    // NMS (skip if end2end since model already applied NMS)
    if (resolved_ != Resolved::END2END_6) {
        dets = Utils::applyNMS(dets, iouThreshold, agnosticNMS);
    }

    return dets;
}

// ============================================================
//  RAW_CxN: tensor shape [1, 4+C, N]
//  rows = features (cx,cy,w,h,cls0,cls1,...), cols = anchors
// ============================================================
std::vector<Detection> OutputParser::parseRawCxN(const cv::Mat& out, float conf) const {
    // Reshape to [rows, cols] = [4+C, N]
    int rows = out.size[1];  // 4 + numClasses
    int cols = out.size[2];  // num predictions

    // We interpret: for each anchor column j:
    //   cx = data[0*cols + j], cy = data[1*cols + j], w = .., h = ..
    //   class scores start at row 4
    const float* data = reinterpret_cast<const float*>(out.data);

    int effectiveC = (numClasses_ > 0) ? numClasses_ : (rows - 4);

    std::vector<Detection> dets;
    for (int j = 0; j < cols; ++j) {
        // Find max class score
        float maxScore = -1e9f;
        int   maxClass = 0;
        for (int c = 0; c < effectiveC; ++c) {
            float s = data[(4 + c) * cols + j];
            if (s > maxScore) { maxScore = s; maxClass = c; }
        }
        if (maxScore < conf) continue;

        float cx = data[0 * cols + j];
        float cy = data[1 * cols + j];
        float w  = data[2 * cols + j];
        float h  = data[3 * cols + j];

        // cx,cy,w,h in letterboxed pixel space → normalise by imgSize
        // (OpenCV DNN blob is [1,3,H,W] with H=W=imgSize; coordinates from model
        //  are already relative to imgSize unless model outputs absolute)
        // YOLO models after sigmoid output relative [0,1] coords; raw onnx may not
        // have sigmoid. We check: if values >> 1 they are absolute pixel coords.
        Detection d;
        d.confidence = maxScore;
        d.classId    = maxClass;

        // If coords seem absolute (>1 when we expect [0,1]):
        // We'll store as normalised – downstream code multiplies by imgSize
        d.boxNorm.x      = cx;      // will be normalised later if needed
        d.boxNorm.y      = cy;
        d.boxNorm.width  = w;
        d.boxNorm.height = h;
        dets.push_back(d);
    }
    return dets;
}

// ============================================================
//  RAW_NxC: tensor shape [1, N, 4+C]
//  rows = anchors, cols = features
// ============================================================
std::vector<Detection> OutputParser::parseRawNxC(const cv::Mat& out, float conf) const {
    int N    = out.size[1];
    int cols = out.size[2];
    int effectiveC = (numClasses_ > 0) ? numClasses_ : (cols - 4);

    const float* data = reinterpret_cast<const float*>(out.data);
    std::vector<Detection> dets;

    for (int i = 0; i < N; ++i) {
        const float* row = data + i * cols;
        float maxScore = -1e9f;
        int   maxClass = 0;
        for (int c = 0; c < effectiveC; ++c) {
            if (row[4 + c] > maxScore) { maxScore = row[4 + c]; maxClass = c; }
        }
        if (maxScore < conf) continue;

        Detection d;
        d.confidence     = maxScore;
        d.classId        = maxClass;
        d.boxNorm.x      = row[0];
        d.boxNorm.y      = row[1];
        d.boxNorm.width  = row[2];
        d.boxNorm.height = row[3];
        dets.push_back(d);
    }
    return dets;
}

// ============================================================
//  OBJ_NxC: tensor shape [1, N, 5+C]
//  rows = anchors, col layout: cx, cy, w, h, objectness, cls0...
// ============================================================
std::vector<Detection> OutputParser::parseObjNxC(const cv::Mat& out, float conf) const {
    int N    = out.size[1];
    int cols = out.size[2];
    int effectiveC = (numClasses_ > 0) ? numClasses_ : (cols - 5);

    const float* data = reinterpret_cast<const float*>(out.data);
    std::vector<Detection> dets;

    for (int i = 0; i < N; ++i) {
        const float* row = data + i * cols;
        float objConf = row[4];
        if (objConf < conf) continue;

        float maxCls = -1e9f;
        int   maxClass = 0;
        for (int c = 0; c < effectiveC; ++c) {
            if (row[5 + c] > maxCls) { maxCls = row[5 + c]; maxClass = c; }
        }
        float score = objConf * maxCls;
        if (score < conf) continue;

        Detection d;
        d.confidence     = score;
        d.classId        = maxClass;
        d.boxNorm.x      = row[0];
        d.boxNorm.y      = row[1];
        d.boxNorm.width  = row[2];
        d.boxNorm.height = row[3];
        dets.push_back(d);
    }
    return dets;
}

// ============================================================
//  END2END_6: tensor shape [1, N, 6]
//  x1, y1, x2, y2, score, class_id  (NMS already done)
// ============================================================
std::vector<Detection> OutputParser::parseEnd2End6(const cv::Mat& out, float conf) const {
    int N = out.size[1];
    const float* data = reinterpret_cast<const float*>(out.data);
    std::vector<Detection> dets;

    for (int i = 0; i < N; ++i) {
        const float* row = data + i * 6;
        float x1 = row[0], y1 = row[1], x2 = row[2], y2 = row[3];
        float score   = row[4];
        int   classId = static_cast<int>(row[5]);

        if (score < conf) continue;
        if (x2 <= x1 || y2 <= y1) continue;

        Detection d;
        d.confidence     = score;
        d.classId        = classId;
        // Store as cx,cy,w,h normalised (inverse letterbox uses this)
        d.boxNorm.x      = x1;          // will be treated as x1 by inverseLetterbox
        d.boxNorm.y      = y1;
        d.boxNorm.width  = x2 - x1;
        d.boxNorm.height = y2 - y1;
        dets.push_back(d);
    }
    return dets;
}
