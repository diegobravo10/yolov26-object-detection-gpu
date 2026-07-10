#pragma once
#include "Utils.hpp"
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>

// ============================================================
//  OutputParser: detect and parse YOLO DNN output tensors
// ============================================================

class OutputParser {
public:
    enum class Format {
        AUTO,
        RAW,      // [1,4+C,N] or [1,N,4+C]  – no objectness
        END2END   // [1,N,6]  x1,y1,x2,y2,score,class_id (NMS done inside)
    };

    // Detected sub-format (resolved from AUTO)
    enum class Resolved {
        UNKNOWN,
        RAW_CxN,   // [1, 4+C, N] – columns are anchors
        RAW_NxC,   // [1, N, 4+C] – rows are anchors
        OBJ_NxC,   // [1, N, 5+C] – with objectness score
        END2END_6  // [1, N, 6]   – x1,y1,x2,y2,score,class_id
    };

    OutputParser(Format fmt, int numClasses);

    // Call once after first net.forward() to print tensor info & resolve AUTO
    // Returns false if format remains ambiguous and user must specify --output-format
    bool inspect(const std::vector<cv::Mat>& outputs,
                 const std::vector<std::string>& outLayerNames);

    // Main parse entry: returns detections in normalised [0,1] letterbox coords
    std::vector<Detection> parse(const std::vector<cv::Mat>& outputs,
                                  float confThreshold,
                                  float iouThreshold,
                                  bool  agnosticNMS) const;

    Resolved  resolved()   const { return resolved_; }
    int       numClasses() const { return numClasses_; }
    std::string formatName() const;

private:
    Format   format_;
    Resolved resolved_{Resolved::UNKNOWN};
    int      numClasses_;
    bool     inspected_{false};

    // Sub-parsers
    std::vector<Detection> parseRawCxN  (const cv::Mat& out, float conf) const;
    std::vector<Detection> parseRawNxC  (const cv::Mat& out, float conf) const;
    std::vector<Detection> parseObjNxC  (const cv::Mat& out, float conf) const;
    std::vector<Detection> parseEnd2End6(const cv::Mat& out, float conf) const;

    Resolved autoResolve(const cv::Mat& out) const;
};
