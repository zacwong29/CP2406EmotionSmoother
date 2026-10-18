//
// Created by zappe on 29/09/2026.
//
#pragma once

#include <map>
#include <string>
#include <vector>
#ifndef AUDIO_MIC_ONNX_EMOTION_SMOOTHER_H
#define AUDIO_MIC_ONNX_EMOTION_SMOOTHER_H

struct windowPrediction {
    float startTime;
    float endTime;

    std::string firstStrongestEmotion;
    float firstStrongestProbability;

    std::string secondStrongestEmotion;
    float secondStrongestProbability;
};

struct smoothedEmotion {
    std::string winningEmotion;
    float confidence;
    std::map<std::string, float> scoreInformation;
};

class EmotionSmoother {
public:
    std::vector<windowPrediction> analyseWindows(std::vector<float> audio, int sampleRate, float windowSeconds, float overlapSeconds);
    smoothedEmotion smoothPredictions(const std::vector<windowPrediction>& windows);
};

#endif //AUDIO_MIC_ONNX_EMOTION_SMOOTHER_H
