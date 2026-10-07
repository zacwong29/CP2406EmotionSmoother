#include "audio_recorder.h"
#include <portaudio.h>
#include <sndfile.h>
#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace {
void check(PaError e, const char* action) {
    if (e != paNoError) throw std::runtime_error(std::string(action) + ": " + Pa_GetErrorText(e));
}
struct PaGuard {
    PaGuard() { check(Pa_Initialize(), "PortAudio initialisation failed"); }
    ~PaGuard() { Pa_Terminate(); }
};
}

void AudioRecorder::printInputDevices() {
    PaGuard guard;
    const int count = Pa_GetDeviceCount();
    std::cout << "Available microphone devices:\n";
    for (int i = 0; i < count; ++i) {
        const PaDeviceInfo* d = Pa_GetDeviceInfo(i);
        if (d && d->maxInputChannels > 0)
            std::cout << "  " << i << ": " << d->name << "\n";
    }
}

std::vector<float> AudioRecorder::record(float seconds) const {
    if (seconds <= 0.0f || seconds > 30.0f)
        throw std::runtime_error("Recording duration must be between 0 and 30 seconds.");
    PaGuard guard;
    PaStream* stream = nullptr;
    PaStreamParameters input{};
    input.device = Pa_GetDefaultInputDevice();
    if (input.device == paNoDevice) throw std::runtime_error("No default microphone was found.");
    const PaDeviceInfo* info = Pa_GetDeviceInfo(input.device);
    input.channelCount = 1;
    input.sampleFormat = paFloat32;
    input.suggestedLatency = info->defaultLowInputLatency;
    input.hostApiSpecificStreamInfo = nullptr;
    check(Pa_OpenStream(&stream, &input, nullptr, sampleRate, 256, paClipOff, nullptr, nullptr),
          "Could not open microphone");
    const size_t total = static_cast<size_t>(seconds * sampleRate);
    std::vector<float> audio(total, 0.0f);
    try {
        check(Pa_StartStream(stream), "Could not start microphone");
        std::cout << "Recording now... speak naturally.\n";
        size_t done = 0;
        while (done < total) {
            const unsigned long block = static_cast<unsigned long>(std::min<size_t>(256, total - done));
            check(Pa_ReadStream(stream, audio.data() + done, block), "Microphone read failed");
            done += block;
        }
        check(Pa_StopStream(stream), "Could not stop microphone");
        Pa_CloseStream(stream);
    } catch (...) {
        if (stream) { Pa_AbortStream(stream); Pa_CloseStream(stream); }
        throw;
    }
    std::cout << "Recording complete.\n";
    return audio;
}

void AudioRecorder::saveWav(const std::string& path, const std::vector<float>& samples) {
    SF_INFO info{}; info.samplerate = sampleRate; info.channels = 1; info.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;
    SNDFILE* file = sf_open(path.c_str(), SFM_WRITE, &info);
    if (!file) throw std::runtime_error("Could not create " + path);
    sf_write_float(file, samples.data(), static_cast<sf_count_t>(samples.size()));
    sf_close(file);
}

std::vector<float> AudioRecorder::loadWaveFile(
    const std::string& filename,
    int& sampleRate)
{
    SF_INFO info{};

    SNDFILE* file =
        sf_open(
            filename.c_str(),
            SFM_READ,
            &info);

    if (!file)
    {
        throw std::runtime_error(
            "Could not open WAV file: " +
            filename);
    }

    sampleRate = info.samplerate;

    std::vector<float> interleaved(
        static_cast<size_t>(
            info.frames *
            info.channels));

    sf_read_float(
        file,
        interleaved.data(),
        static_cast<sf_count_t>(
            interleaved.size()));

    sf_close(file);

    std::vector<float> mono(
        static_cast<size_t>(
            info.frames));

    for (sf_count_t i = 0;
         i < info.frames;
         ++i)
    {
        float sum = 0.0f;

        for (int c = 0;
             c < info.channels;
             ++c)
        {
            sum += interleaved[
                static_cast<size_t>(
                    i * info.channels + c)];
        }

        mono[
            static_cast<size_t>(i)] =
            sum /
            static_cast<float>(
                info.channels);
    }

    std::cout
        << "Loaded WAV\n"
        << "Sample Rate : "
        << info.samplerate
        << "\nChannels    : "
        << info.channels
        << "\nFrames      : "
        << info.frames
        << "\n";

    return mono;
}
