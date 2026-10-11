//
// Created by zappe on 29/09/2026.
//
#include "emotion_smoother.h"
#include <iostream>
#include <vector>

std::vector<float> EmotionSmoother::analyseWindows(std::vector<float> audio, int sampleRate, float windowSeconds, float overlapSeconds) {
    if (audio.empty()) {
        throw std::invalid_argument("Audio cannot be empty.");
    }

    if (sampleRate <= 0) {
        throw std::invalid_argument("Sample rate must be positive.");
    }

    if (windowSeconds <= 0.0F) {
        throw std::invalid_argument("Window length must be positive.");
    }

    if (overlapSeconds < 0.0F || overlapSeconds > windowSeconds) {
        throw std::invalid_argument(
            "Overlap must be at least zero and smaller than the window length.");
    }

    const auto windowSamples = static_cast<std::size_t>(
        std::lround(windowSeconds * sampleRate));

    const auto overlapSamples = static_cast<std::size_t>(
        std::lround(overlapSeconds * sampleRate));

    const auto stepSamples = windowSamples - overlapSamples;

    if (windowSamples < 1600) {
        throw std::invalid_argument(
            "Each window must contain at least 1,600 samples.");
    }
}
