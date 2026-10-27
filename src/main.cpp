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
#include <string>
#include <vector>

// TODO (Task 3): Convert the one-run menu flow into a loop so another
// recording or WAV file can be analysed without restarting.
int main()
{


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

                    std::string command =
                        "ffmpeg -y -i \"" + wavFile +
                        "\" -ar 16000 -ac 1 temp_16k.wav";

                    std::system(command.c_str());

                    audio = recorder.loadWaveFile(
                        "temp_16k.wav",
                        sampleRate);
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