#include "emotion_model.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <stdexcept>

static std::vector<std::string> loadLabels(const std::string& path) {
    std::ifstream f(path); if (!f) throw std::runtime_error("Cannot open labels: " + path);
    std::vector<std::string> out; for (std::string s; std::getline(f,s);) if(!s.empty()) out.push_back(s);
    return out;
}
static void normalise(std::vector<float>& x) {
    const double mean=std::accumulate(x.begin(),x.end(),0.0)/x.size(); double v=0;
    for(float s:x){const double d=s-mean;v+=d*d;} const double sd=std::sqrt(v/x.size()+1e-7);
    for(float& s:x)s=static_cast<float>((s-mean)/sd);
}

EmotionModel::EmotionModel(const std::string& modelPath, const std::string& labelsPath)
    : env(ORT_LOGGING_LEVEL_WARNING,"speech-emotion"), labels(loadLabels(labelsPath)) {
    options.SetIntraOpNumThreads(1); options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
#ifdef _WIN32
    std::wstring w(modelPath.begin(),modelPath.end()); session=std::make_unique<Ort::Session>(env,w.c_str(),options);
#else
    session=std::make_unique<Ort::Session>(env,modelPath.c_str(),options);
#endif
}

std::vector<Prediction> EmotionModel::predict(std::vector<float> waveform) {
    if (waveform.size() < 1600) throw std::runtime_error("Recording is too short.");
    normalise(waveform);
    Ort::AllocatorWithDefaultOptions allocator;
    const size_t n=session->GetInputCount();
    std::vector<std::string> namesStorage; std::vector<const char*> names;
    for(size_t i=0;i<n;++i){auto p=session->GetInputNameAllocated(i,allocator);namesStorage.emplace_back(p.get());}
    for(auto& s:namesStorage)names.push_back(s.c_str());
    std::vector<int64_t> shape{1,static_cast<int64_t>(waveform.size())};
    auto memory=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
    std::vector<Ort::Value> inputs; inputs.reserve(n);
    std::vector<int64_t> mask(waveform.size(),1);
    for(const auto& name:namesStorage) {
        if(name.find("mask")!=std::string::npos)
            inputs.emplace_back(Ort::Value::CreateTensor<int64_t>(memory,mask.data(),mask.size(),shape.data(),shape.size()));
        else
            inputs.emplace_back(Ort::Value::CreateTensor<float>(memory,waveform.data(),waveform.size(),shape.data(),shape.size()));
    }
    auto outAllocated=session->GetOutputNameAllocated(0,allocator); const char* outName=outAllocated.get();
    auto outputs=session->Run(Ort::RunOptions{nullptr},names.data(),inputs.data(),inputs.size(),&outName,1);
    const size_t count=outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
    if(count!=labels.size()) throw std::runtime_error("Model output and labels.txt have different class counts.");
    const float* logits=outputs[0].GetTensorData<float>();
    const float maximum=*std::max_element(logits,logits+count); std::vector<float> probabilities(count); double sum=0;
    for(size_t i=0;i<count;++i){probabilities[i]=std::exp(logits[i]-maximum);sum+=probabilities[i];}
    std::vector<Prediction> result;
    for(size_t i=0;i<count;++i)result.push_back({labels[i],static_cast<float>(probabilities[i]/sum)});
    std::sort(result.begin(),result.end(),[](const auto& a,const auto& b){return a.probability>b.probability;});
    return result;
}
