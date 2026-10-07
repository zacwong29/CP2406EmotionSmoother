// Keep the supplied recording and WAV-loading pathway working.
// TODO (Task 1): Use sampleRate when converting window seconds to samples.
#pragma once
#include <string>
#include <vector>

class AudioRecorder {
public:
    static constexpr int sampleRate = 16000;
    std::vector<float> record(float seconds) const;
    static void saveWav(const std::string& path, const std::vector<float>& samples);
    static void printInputDevices();
    std::vector<float> loadWaveFile(const std::string& filename, int& sampleRate);
};

