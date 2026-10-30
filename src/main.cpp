// ============================================================
// CP2406 Assessment Task 3: Student implementation points
//
// TODO (Task 1): Add window-based analysis. Divide audio into
// valid consecutive or overlapping windows, convert seconds to
// samples at 16,000 Hz, skip windows shorter than 1,600 samples,
// and retain the top two labels and probabilities with times.
//
// TODO (Task 2): Implement emotion smoothing. Combine top and
// second predictions using rank and probability weights, handle
// empty input, calculate confidence, and resolve ties predictably.
// As required by the assignment, students should create
// EmotionSmoother.h and EmotionSmoother.cpp themselves.
//
// TODO (Task 3): Display the window timeline and final smoothed
// result. Compare the same audio using at least three window
// lengths and permit repeated analysis without restarting.
//
// TODO (Task 4): Refactor main() into cohesive helper functions,
// use meaningful names, const where appropriate, safe indexing,
// and clear exception handling. Complete the required code-review
// responses, tests, and AI-use statement in the submitted report.
// ============================================================

#include "audio_recorder.h"
#include "emotion_model.h"
#include "emotion_smoother.h"
#include <onnxruntime_cxx_api.h>
#include <iomanip>
#include <iostream>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

void testSmoothPredictions() {
    std::vector<windowPrediction> testWindows = {
        {
            0.0f, 1.0f,
            "calm", 0.70f,
            "happy", 0.20f
        },
        {
            1.0f, 2.0f,
            "calm", 0.65f,
            "sad", 0.25f
        },
        {
        2.0f, 3.0f,
            "calm", 0.55f,
            "happy", 0.35f
        }
    };

    EmotionSmoother smoother;

    auto result = smoother.smoothPredictions(testWindows);

    std::cout << "\n=== MANUAL SMOOTHER TEST ===\n";
    std::cout << "Winner: "
              << result.winningEmotion
              << '\n';

    std::cout << "Confidence: "
              << result.confidence
              << '\n';
}

void testEmptyInput()
{
    std::vector<windowPrediction> windows;

    EmotionSmoother smoother;

    try
    {
        auto result = smoother.smoothPredictions(windows);

        std::cout << "ERROR: exception expected\n";
    }
    catch (const std::exception&)
    {
        std::cout << "Empty-input test passed\n";
    }
}

void testSingleWindow()
{
    std::vector<windowPrediction> windows = {
        {
            0.0f, 1.0f,
            "happy", 0.8f,
            "sad", 0.2f
        }
    };

    EmotionSmoother smoother;

    auto result = smoother.smoothPredictions(windows);

    std::cout
        << "Single window winner = "
        << result.winningEmotion
        << '\n';
}

void testRepeatedTopEmotion()
{
    std::vector<windowPrediction> windows = {
        {
            0.0f, 1.0f,
            "happy", 0.8f,
            "sad", 0.2f
        },
        {
            1.0f, 2.0f,
            "happy", 0.7f,
            "angry", 0.3f
        },
        {
            2.0f, 3.0f,
            "happy", 0.9f,
            "calm", 0.1f
        }
    };

    EmotionSmoother smoother;

    auto result = smoother.smoothPredictions(windows);

    std::cout
        << "Repeated-top winner = "
        << result.winningEmotion
        << '\n';
}

void testTieBreak()
{
    std::vector<windowPrediction> windows = {
        {
            0.0f, 1.0f,
            "angry", 1.0f,
            "happy", 1.0f
        },
        {
            1.0f, 2.0f,
            "happy", 1.0f,
            "angry", 1.0f
        }
    };

    EmotionSmoother smoother;

    auto result = smoother.smoothPredictions(windows);

    std::cout
        << "Tie-break winner = "
        << result.winningEmotion
        << '\n';
}


// TODO (Task 3): Convert the one-run menu flow into a loop so another
// recording or WAV file can be analysed without restarting.
int main()
{

    testSmoothPredictions();
    testEmptyInput();
    testSingleWindow();
    testRepeatedTopEmotion();
    testTieBreak();


    try {
        int choice = 0;
        while (choice != 4) {
            std::cout
                << "=====================================\n"
                << " Speech Emotion Recognition\n"
                << "=====================================\n\n"
                << "1. Record from microphone\n"
                << "2. Load WAV file\n"
                << "3. List microphone devices\n"
                << "4. Quit\n\n"
                << "Choice: ";

            if (!(std::cin >> choice)) {
                std::cin.clear();
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                std::cout << "Please enter a number from 1 to 4.\n";
                continue;
            }

            AudioRecorder recorder;
            std::cin.ignore();

            if (choice == 4)
            {
                break;
            }

            if (choice == 3)
            {
                AudioRecorder::printInputDevices();
                continue;
            }

            std::vector<float> audio;
            bool convertedFileCreated = false;

            if (choice == 1)
            {
                float seconds;

                std::cout
                    << "\nRecording duration (seconds): ";

                std::cin >> seconds;
                std::cin.ignore();

                std::cout
                    << "\nPress ENTER and start speaking...\n";

                std::cin.get();

                audio = recorder.record(seconds);

                AudioRecorder::saveWav(
                    "last_recording.wav",
                    audio);


                std::cout
                    << "\nSaved recording to:\n"
                    << "last_recording.wav\n\n";
            }
            else if (choice == 2)
            {
                std::string wavFile;

                std::cout
                    << "\nEnter WAV filename:\n";

                std::getline(
                    std::cin,
                    wavFile);

                int sampleRate;

                audio = recorder.loadWaveFile(
                    wavFile,
                    sampleRate);


                if (sampleRate != AudioRecorder::sampleRate)
                {
                    std::cout
                        << "Converting "
                        << sampleRate
                        << " Hz audio to 16000 Hz...\n";

                    const std::string tempFile = "temp_16k.wav";

                    std::string command =
                        "ffmpeg -y -i \"" + wavFile +
                        "\" -ar 16000 -ac 1 \"" + tempFile + "\"";

                    int result = std::system(command.c_str());

                    if (result != 0) {
                        throw std::runtime_error(
                            "Audio conversion failed. "
                            "Ensure FFmpeg is installed and the WAV file is valid."
                        );
                    }

                    audio = recorder.loadWaveFile(
                        tempFile,
                        sampleRate);

                    if (sampleRate != AudioRecorder::sampleRate) {
                        throw std::runtime_error(
                            "Converted audio is not 16000 Hz."
                        );
                    }

                    convertedFileCreated = true;
                }

                std::cout
                    << "\nLoaded: "
                    << wavFile
                    << "\n";
            }
            else
            {
                std::cout
                    << "Invalid option.\n";

                continue;
            }

            EmotionModel model(
                "model/model.onnx",
                "model/labels.txt");

            auto predictions =
                model.predict(audio);

            EmotionSmoother smoother;

            auto windows = smoother.analyseWindows(
                audio,
                AudioRecorder::sampleRate,
                1.0f,
                0.5f
            );

            for (const auto& window : windows) {
                std::cout << std::fixed << std::setprecision(1)
                          << window.startTime << "-"
                          << window.endTime << " s: "
                          << window.firstStrongestEmotion << " "
                          << window.firstStrongestProbability * 100.0F << "% | "
                          << window.secondStrongestEmotion << " "
                          << window.secondStrongestProbability * 100.0F << "%\n";
            }

            std::cout << "\nWindow count: "
                      << windows.size()
                      << "\n";

            std::cout
                << "\n=====================================\n"
                << " Emotion Probabilities\n"
                << "=====================================\n\n";

            for (const auto& prediction : predictions)
            {
                std::cout
                    << std::setw(12)
                    << prediction.label
                    << " : "
                    << std::fixed
                    << std::setprecision(2)
                    << prediction.probability * 100.0f
                    << "%\n";
            }

            std::cout
                << "\n=====================================\n"
                << " Predicted Emotion\n"
                << "=====================================\n\n"
                << predictions.front().label
                << "\n";

            if (convertedFileCreated) {
                std::remove("temp_16k.wav");
            }

        }
        }
        catch (const Ort::Exception& e)
        {
            std::cerr
                << "\nONNX Runtime Error:\n"
                << e.what()
                << "\n";

            return 1;
        }
        catch (const std::exception& e)
        {
            std::cerr
                << "\nError:\n"
                << e.what()
                << "\n";

            return 1;
        }
    return 0;
}