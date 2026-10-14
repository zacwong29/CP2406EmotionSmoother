//
// Created by zappe on 29/09/2026.
//
#include "emotion_smoother.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "emotion_model.h"

std::vector<windowPrediction> EmotionSmoother::analyseWindows(std::vector<float> audio, int sampleRate, float windowSeconds, float overlapSeconds) {
    if (audio.empty()) {
        throw std::invalid_argument("Audio cannot be empty.");
    }

    if (sampleRate <= 0) {
        throw std::invalid_argument("Sample rate must be positive.");
    }

    if (windowSeconds <= 0.0F) {
        throw std::invalid_argument("Window length must be positive.");
    }

    if (overlapSeconds < 0.0F || overlapSeconds >= windowSeconds) {
        throw std::invalid_argument(
            "Overlap must be at least zero and smaller than the window length.");
    }

    const auto windowSamples = static_cast<std::size_t>(
        std::lround(windowSeconds * sampleRate));

    const auto overlapSamples = static_cast<std::size_t>(
        std::lround(overlapSeconds * sampleRate));

    if (overlapSamples >= windowSamples) {
        throw std::invalid_argument(
            "Overlap in samples must be smaller than the window.");
    }

    const auto stepSamples = windowSamples - overlapSamples;

    if (windowSamples < 1600) {
        throw std::invalid_argument(
            "Each window must contain at least 1,600 samples.");
    }

    std::vector<windowPrediction> windows;

    EmotionModel model(
    "model/model.onnx",
    "model/labels.txt");

    for (std::size_t start = 0;
        start < audio.size();
        start += stepSamples) {

        const std::size_t end = std::min(
            start + windowSamples,
            audio.size());

        if (end - start < 1600) {
            break;
        }

        std::vector<float> segment(
            audio.begin() + static_cast<std::ptrdiff_t>(start),
            audio.begin() + static_cast<std::ptrdiff_t>(end));

        auto predictions = model.predict(std::move(segment));

        if (predictions.size() < 2) {
            throw std::runtime_error(
                "The model must return at least two emotion classes.");
        }

        windowPrediction result {
            static_cast<float>(start) / sampleRate,
            static_cast<float>(end) / sampleRate,
            predictions[0].label,
            predictions[0].probability,
            predictions[1].label,
            predictions[1].probability
        };

        windows.push_back(std::move(result));


        if (end == audio.size()) {
            break;
        }
    }

    return windows;
}
