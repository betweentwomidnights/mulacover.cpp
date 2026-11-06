// Adapted from betweentwomidnights/yue2.cpp (MIT); see LICENSES/YuE2-MIT.txt.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace mulacover::server {

std::string base64_encode(const std::uint8_t * data, std::size_t size);

// Decodes standard padded base64. ASCII whitespace is ignored and a leading
// `data:<type>;base64,` prefix is accepted. Throws std::invalid_argument.
std::vector<std::uint8_t> base64_decode(std::string_view text);

} // namespace mulacover::server
