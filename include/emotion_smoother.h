//
// Created by zappe on 29/09/2026.
//
#include <vector>
#ifndef AUDIO_MIC_ONNX_EMOTION_SMOOTHER_H
#define AUDIO_MIC_ONNX_EMOTION_SMOOTHER_H

class EmotionSmoother {
public:
    std::vector<float> analyseWindows(std::vector<float> audio, int sampleRate, float windowSeconds, float overlapSeconds);
};

#endif //AUDIO_MIC_ONNX_EMOTION_SMOOTHER_H
