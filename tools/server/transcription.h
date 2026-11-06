#pragma once
#include "service.h"
namespace mulacover::server {
struct TranscribedScore { std::string melody,chords;json::Value metadata=json::object(); };
using Exchange=std::function<HttpResponse(const HttpRequest &)>;
// Transcribe exactly the borrowed context window; future PCM is never sent.
// Only this request's provider session is polled/cancelled/consumed.
TranscribedScore transcribe_score(const Config &,const wave_io::Wave &,int64_t start,int64_t count,
    const std::string & lane,const Progress &,const Cancel &,Exchange exchange={});
}
