#pragma once
#include "http.h"
#include "json.h"
#include "../wave-io.h"
#include "mulacover_job.h"
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>

namespace mulacover::server {
struct Config {
    std::string model_dir="models", codec_model_dir="models", backend="cuda";
    int threads=4, max_cache_capacity=2048;
    int port=8010;
    int transcription_port=8011;
    std::string allow_origin;
    std::string tokenizers, style, encoder, generator, flow, sq;
    void resolve_models();
};
struct Input {
    wave_io::Wave audio;
    json::Value options;
    std::string melody, chords, drums; // uploaded MIDI bytes, never caller paths
};
struct Outcome { std::string wav, full_wav; json::Value meta=json::object(); };
using Progress=std::function<void(double,const std::string &,int64_t,int64_t)>;
using Cancel=std::function<bool()>;
using Runner=std::function<Outcome(const Config &,const Input &,const Progress &,const Cancel &)>;
Outcome run_native(const Config &,const Input &,const Progress &,const Cancel &);
mulacover_job_plan_v1 plan_audio_input(const Config &,const Input &);
std::string props(const Config &);
void resolve_job_seeds(json::Value &); // -1 resolves once, at job admission

// One accepted GPU job at a time. Terminal results expire after 30 minutes;
// at most four are retained. HTTP callbacks never own model contexts.
class Service {
public:
    explicit Service(Config config,Runner runner=run_native);
    ~Service();
    HttpResponse handle(const HttpRequest &);
    void shutdown(); // cancel/join the worker, releasing models before return
private:
    struct Job;
    Config config_;
    Runner runner_;
    std::mutex mutex_;
    std::condition_variable wake_;
    std::map<std::string,std::shared_ptr<Job>> jobs_;
    std::shared_ptr<Job> pending_;
    bool stopping_=false, busy_=false;
    std::thread worker_;
    void work();
    void prune(); // mutex held
    HttpResponse route(const HttpRequest &);
};
}
