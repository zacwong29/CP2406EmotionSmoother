// Keep the supplied model prediction pathway unchanged.
// TODO (Task 4): In your code review, explain pass-by-value waveform
// normalisation and why returned predictions are sorted.
#pragma once
#include <memory>
#include <string>
#include <vector>
#include <onnxruntime_cxx_api.h>

struct Prediction { std::string label; float probability; };
class EmotionModel {
public:
    EmotionModel(const std::string& modelPath, const std::string& labelsPath);
    std::vector<Prediction> predict(std::vector<float> waveform);
private:
    Ort::Env env;
    Ort::SessionOptions options;
    std::unique_ptr<Ort::Session> session;
    std::vector<std::string> labels;
};
