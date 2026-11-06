// Adapted from betweentwomidnights/yue2.cpp (MIT); see LICENSES/YuE2-MIT.txt.
#pragma once

#include <optional>
#include <string>
#include <vector>

namespace mulacover::server {

struct MultipartPart {
    std::string name;
    std::string filename;
    std::string content_type;
    std::string data;
};

std::optional<std::string> multipart_boundary(const std::string & content_type);
std::vector<MultipartPart> parse_multipart(
    const std::string & body,
    const std::string & boundary);

} // namespace mulacover::server
