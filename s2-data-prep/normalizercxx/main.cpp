#include <iostream>
#include <vector>
#include <span>
#include <stdexcept>
#include <format>
#include <thread>
#include <mutex>
#include <cmath>
#include <memory>
#include <optional>
#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;

#include <sndfile.h>
#include <samplerate.h>



constexpr int WindowSize = 2048;
constexpr int MP3SmpRate = 48000;
constexpr int OutputSampleRate = 16000;
constexpr bool ShouldNormalize = false;
thread_local std::atomic_bool DebugMode = false;


struct SFCloser {void operator()(SNDFILE* sf){sf_close(sf);}};

std::optional<std::vector<float>> readmp3file(const std::string& filename)
{
    SF_INFO info;
    std::unique_ptr<SNDFILE, SFCloser> sf(sf_open(filename.c_str(), SFM_READ, &info));

    if(!sf || info.channels > 1 || info.frames < 0) {
        return std::nullopt;
    }

    std::vector<float> ret(info.frames);
    sf_read_float(sf.get(), ret.data(), info.frames);
    return ret;
}

std::vector<float> resample(const std::span<float> inSig, int inputRate, int outputRate)
{
    double rateRatio = double(outputRate) / double(inputRate);
    long inFrames = inSig.size();
    long outFrames = rateRatio * inFrames;
    std::vector<float> ret(outFrames);

    SRC_DATA srcData {
        .data_in = inSig.data(),
        .data_out = ret.data(),
        .input_frames = inFrames,
        .output_frames = outFrames,
        .src_ratio = rateRatio
    };

    src_simple(&srcData, SRC_SINC_FASTEST, 1);
    return ret;
}

void savewavfile(const std::string& filename, const std::span<float> signal, int rate)
{
    SF_INFO info;
    info.channels = 1;
    info.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;
    info.samplerate = rate;

    std::unique_ptr<SNDFILE, SFCloser> sf(sf_open(filename.c_str(), SFM_WRITE, &info));

    sf_write_float(sf.get(), signal.data(), signal.size());
}

template<typename T>
constexpr std::span<T> window_subspan(const std::span<T> span, int maxSize, int offset)
{
    if(offset >= span.size()) {
        return span.last(0);
    } else if(offset + maxSize >= span.size()) {
        return span.subspan(offset);
    } else {
        return span.subspan(offset, maxSize);
    }
}

template<typename T>
constexpr std::span<T> clamped_subspan(const std::span<T> span, int begin, int end)
{
    begin = std::max(0, begin);
    end = std::min<int>(span.size(), end);
    begin = std::min(begin, end);
    return span.subspan(begin, end-begin);
}


void syncPrint(const std::string& msg)
{
    static std::mutex mtx;
    std::lock_guard<std::mutex> lk(mtx);

    std::cout << msg;
}


float rms(const std::span<float> signal)
{
    float ret = 0.0f;
    for(auto& smp: signal) {
        ret += smp * smp;
    }
    ret /= signal.size();
    return std::sqrt(ret);
}

constexpr float dBFS(float x)
{
    if(x == 0.0f) {
        return -300.0f;
    }
    return 20.0f * std::log10(x);
}

constexpr float dBFS_to_ratio(float x)
{
    return std::pow(10.0, x / 20.0);
}

struct SampleStats {
    float lengthSecs;
    float noiseFloor;
    float evrms;
    std::span<float> viableFragment;
};

std::vector<float> winrmsvec(const std::span<float> signal)
{
    std::vector<float> ret;
    ret.reserve(signal.size() / WindowSize);
    for(int i=0; i<signal.size(); i += WindowSize) {
        ret.push_back(rms(window_subspan(signal, WindowSize, i)));
    }
    return ret;
}


constexpr std::span<float> getViableFragment(const std::span<float> signal, /*const std::span<float> rmsvec, */ float noiseFloor)
{
    int first = signal.size() - 1;
    int last = 0;
    int currentRun = 0;
    int numWindows = (signal.size() + WindowSize - 1) / WindowSize;
    for(int i=0; i<numWindows; i++) {
        int posBegin = i * WindowSize;
        int posEnd = posBegin + WindowSize;
        float windowRMS = rms(window_subspan(signal, WindowSize, posBegin));
        if(windowRMS > noiseFloor) {
            // if(DebugMode) {
            //     syncPrint(std::format("viable window at position {}: current run is {} (NF {:.1f} vs RM {:.1f})\n", posBegin, currentRun, dBFS(noiseFloor), dBFS(windowRMS)));
            // }
            currentRun++;
        } else {
            currentRun = 0;
        }
        if(currentRun >= 4) {
            first = std::min(first, posBegin - WindowSize*currentRun);
            last = std::max(last, posEnd);
        } 
    }
    if(first < last) {
        return clamped_subspan(signal, first - 4800, last + 4800);
    } else {
        return signal.subspan(0, 0);
    }
}

SampleStats evrms(const std::span<float> signal)
{
    auto wr = winrmsvec(signal);
    std::sort(wr.begin(), wr.end());
    float noiseFloor = std::max(dBFS_to_ratio(-65.0f), 2.0f * wr[wr.size() / 16]);
    auto lb = std::lower_bound(wr.begin(), wr.end(), noiseFloor);

    float evrms = 1e-15f;
    if(lb != wr.end()) {
        evrms = rms(std::span<float>(wr).subspan(lb - wr.begin()));
    }
    return SampleStats {
        .lengthSecs = signal.size() / (float)MP3SmpRate,
        .noiseFloor = noiseFloor,
        .evrms = evrms,
        .viableFragment = getViableFragment(signal, noiseFloor)
    };
}

std::vector<float> normalize(const std::span<float> sourceData, float dB_actual, float dB_target)
{
    float ratio = std::pow(10.0f, (dB_target - dB_actual) / 20.0);
    std::vector<float> output(sourceData.begin(), sourceData.end());
    for(auto& x: output) {
        x = std::clamp(x*ratio, -0.95f, 0.95f);
    }
    return output;
}


bool shouldReject(const SampleStats& dat)
{
    return 
        dat.lengthSecs > 30.0 ||
        dBFS(dat.evrms) < -50.0f || 
        dBFS(dat.evrms) - dBFS(dat.noiseFloor) < 6.0f ||
        dat.viableFragment.empty();
}

struct FilePair {
    std::string srcFilename;
    std::string dstFilename;
};

std::vector<FilePair> getFileList()
{
    std::vector<FilePair> ret;
    for(int i=0; ; i++) {
        std::string srcFilename = std::format("all/{:06d}.mp3", i);
        if(not fs::exists(srcFilename)) {
            break;
        }
        ret.push_back(FilePair {
            .srcFilename = srcFilename,
            .dstFilename = std::format("wav/{:06d}.wav", i)
        });
    }
    return ret;
}

void processFiles(const std::span<FilePair> files, int threadId = 0)
{
    int i = 0;
    for(auto& filePair: files) {
        try {
            if(auto mp3file = readmp3file(filePair.srcFilename)) {
                // bool enableDebug = (filePair.dstFilename == "wav/000007.wav");
                bool enableDebug = false;
                if(enableDebug) DebugMode = true;
            
                auto dat = evrms(*mp3file);
                bool reject = shouldReject(dat);

                if(!reject) {
                    if constexpr(ShouldNormalize) {
                        auto normalized = normalize(dat.viableFragment, dBFS(dat.evrms), -20.0f);
                        auto resampled = resample(normalized, MP3SmpRate, OutputSampleRate);
                        savewavfile(filePair.dstFilename, resampled, OutputSampleRate);
                    } else {
                        auto resampled = resample(*mp3file, MP3SmpRate, OutputSampleRate);
                        savewavfile(filePair.dstFilename, resampled, OutputSampleRate);
                    }
                }

                if(enableDebug) DebugMode = false;

                if((i++) % 64 == 0) syncPrint(std::format("[thr #{}] [{}%] {}: EVRMS = {:.1f} dBFS, NF = {:.1f} dBFS {}\n", 
                    threadId,
                    (long long)(i * 100) / (files.size()),
                    filePair.srcFilename, 
                    dBFS(dat.evrms), 
                    dBFS(dat.noiseFloor),
                    (reject ? "REJECT" : "")
                ));

            } else {
                continue;
            }
        } catch(std::exception& ex) {
            std::cerr << std::format("exception at file {}: {}", filePair.srcFilename, ex.what());
            //std::terminate();
        }
    }
}

int main(int, char**) 
{
    std::cout << "getting file list...\n";
    auto fileList = getFileList();

    std::vector<std::jthread> threads;

    int numThreads = std::thread::hardware_concurrency();
    int workload = (fileList.size() + numThreads - 1) / numThreads;
    
    for(int i=0; i<numThreads; i++) {
        int begin = i*workload;
        int size = workload;
        syncPrint(std::format("thread {} gets range from [{} {})\n", i, begin, begin+size));
        threads.emplace_back(std::jthread(processFiles, window_subspan(std::span<FilePair>(fileList), size, begin), i));
    }
}
