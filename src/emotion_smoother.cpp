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
            "Overlap must be smaller than the window length.");
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

    std::cout << "\n=== WINDOW SETTINGS TEST ===\n";
    std::cout << "Window samples: "
              << windowSamples << '\n';

    std::cout << "Overlap samples: "
              << overlapSamples << '\n';

    std::cout << "Step samples: "
              << stepSamples << '\n';

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

        std::cout
    << "Start = " << start
    << ", End = " << end
    << '\n';

        if (end - start < 1600) {
            break;
        }

        std::vector<float> segment(
            audio.begin() + static_cast<std::ptrdiff_t>(start),
            audio.begin() + static_cast<std::ptrdiff_t>(end));

        std::cout
            << "Segment size = "
            << segment.size()
            << '\n';

        auto predictions = model.predict(std::move(segment));

        std::cout
            << "Top emotion = "
            << predictions[0].label
            << " ("
            << predictions[0].probability
            << ")\n";

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

        std::cout
            << "Stored window: "
            << result.startTime
            << "s - "
            << result.endTime
            << "s\n";


        if (end == audio.size()) {
            break;
        }
    }

    std::cout
        << "Total windows analysed: "
        << windows.size()
        << '\n';

    return windows;
}


smoothedEmotion smoothPredictions(const std::vector<windowPrediction>& windows) {
    const float TOP_WEIGHT = 2.0f;  // Constants for clean code.
    const float SECOND_WEIGHT = 1.0f;

    if (TOP_WEIGHT <= SECOND_WEIGHT) {
        throw std::logic_error(
            "Top weight must be greater than second weight.");
    }

    if (windows.empty()) {
        throw std::invalid_argument("Window cannot be empty.");
    }

    std::map<std::string, float> emotionScores;
    std::map<std::string, float> topOnlyScores;

    // Accumulate scores
    for (const auto& window : windows) {
        float topScore = window.firstStrongestProbability * TOP_WEIGHT;
        float secondScore = window.secondStrongestProbability * SECOND_WEIGHT;

        emotionScores[window.firstStrongestEmotion] += topScore;
        emotionScores[window.secondStrongestEmotion] += secondScore;

        topOnlyScores[window.firstStrongestEmotion] += topScore;
    }

    // Find the winning emotion
    std:: string winningEmotion;
    float winningScore = 0.0f;

    for (const auto& [emotion, score] : emotionScores) {
        if (score > winningScore) {  // if scores are equal, the alphabetically first emotion
                                     // is selected due to how maps work in C++ (key order).
            winningScore = score;
            winningEmotion = emotion;
        }
    }

    // Calculate total score
    float totalScore = 0.0f;

    for (const auto& [emotion, score] : emotionScores) {
        totalScore += score;
    }

    float confidence = winningScore / totalScore;

    return {
        winningEmotion,
        confidence,
        emotionScores
    };
}