#pragma once
// Adapted from betweentwomidnights/yue2.cpp (MIT); see LICENSES/YuE2-MIT.txt.
#include "unicode-tables.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
namespace mula::text {
namespace unicode = mula::text_detail;
bool in_ranges(
    std::uint32_t codepoint,
    const unicode::CodepointRange * begin,
    const unicode::CodepointRange * end) {
    const auto found = std::upper_bound(
        begin,
        end,
        codepoint,
        [](std::uint32_t value, const unicode::CodepointRange & range) {
            return value < range.first;
        });
    if (found == begin) return false;
    const auto & range = *(found - 1);
    return codepoint <= range.last;
}

bool is_letter(std::uint32_t codepoint) {
    return in_ranges(
        codepoint,
        std::begin(unicode::letter_ranges),
        std::end(unicode::letter_ranges));
}

bool is_number(std::uint32_t codepoint) {
    return in_ranges(
        codepoint,
        std::begin(unicode::number_ranges),
        std::end(unicode::number_ranges));
}

bool is_whitespace(std::uint32_t codepoint) {
    return (codepoint >= 0x09 && codepoint <= 0x0D) ||
        codepoint == 0x20 || codepoint == 0x85 || codepoint == 0xA0 ||
        codepoint == 0x1680 || (codepoint >= 0x2000 && codepoint <= 0x200A) ||
        codepoint == 0x2028 || codepoint == 0x2029 || codepoint == 0x202F ||
        codepoint == 0x205F || codepoint == 0x3000;
}

bool is_newline(std::uint32_t codepoint) {
    return codepoint == '\r' || codepoint == '\n';
}

std::uint8_t combining_class(std::uint32_t codepoint) {
    const auto begin = std::begin(unicode::combining_ranges);
    const auto end = std::end(unicode::combining_ranges);
    const auto found = std::upper_bound(
        begin,
        end,
        codepoint,
        [](std::uint32_t value, const unicode::CombiningRange & range) {
            return value < range.first;
        });
    if (found == begin) return 0;
    const auto & range = *(found - 1);
    return codepoint <= range.last ? range.value : 0;
}

void append_utf8(std::string & output, std::uint32_t codepoint) {
    if (codepoint <= 0x7F) {
        output.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7FF) {
        output.push_back(static_cast<char>(0xC0 | (codepoint >> 6)));
        output.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    } else if (codepoint <= 0xFFFF) {
        output.push_back(static_cast<char>(0xE0 | (codepoint >> 12)));
        output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
        output.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    } else {
        output.push_back(static_cast<char>(0xF0 | (codepoint >> 18)));
        output.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F)));
        output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
        output.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    }
}

std::vector<std::uint32_t> decode_utf8(const std::string & input) {
    std::vector<std::uint32_t> output;
    output.reserve(input.size());
    for (std::size_t offset = 0; offset < input.size();) {
        const auto first = static_cast<unsigned char>(input[offset]);
        std::uint32_t codepoint = 0;
        std::size_t length = 0;
        std::uint32_t minimum = 0;
        if (first <= 0x7F) {
            codepoint = first;
            length = 1;
        } else if (first >= 0xC2 && first <= 0xDF) {
            codepoint = first & 0x1F;
            length = 2;
            minimum = 0x80;
        } else if (first >= 0xE0 && first <= 0xEF) {
            codepoint = first & 0x0F;
            length = 3;
            minimum = 0x800;
        } else if (first >= 0xF0 && first <= 0xF4) {
            codepoint = first & 0x07;
            length = 4;
            minimum = 0x10000;
        } else {
            throw std::invalid_argument("MuLaCover tokenizer input is not valid UTF-8");
        }
        if (offset + length > input.size()) {
            throw std::invalid_argument("MuLaCover tokenizer input has truncated UTF-8");
        }
        for (std::size_t index = 1; index < length; ++index) {
            const auto byte = static_cast<unsigned char>(input[offset + index]);
            if ((byte & 0xC0) != 0x80) {
                throw std::invalid_argument("MuLaCover tokenizer input is not valid UTF-8");
            }
            codepoint = (codepoint << 6) | (byte & 0x3F);
        }
        if ((length > 1 && codepoint < minimum) || codepoint > 0x10FFFF ||
            (codepoint >= 0xD800 && codepoint <= 0xDFFF)) {
            throw std::invalid_argument("MuLaCover tokenizer input contains an invalid codepoint");
        }
        output.push_back(codepoint);
        offset += length;
    }
    return output;
}

const unicode::Decomposition * find_decomposition(std::uint32_t codepoint) {
    const auto begin = std::begin(unicode::decompositions);
    const auto end = std::end(unicode::decompositions);
    const auto found = std::lower_bound(
        begin,
        end,
        codepoint,
        [](const unicode::Decomposition & entry, std::uint32_t value) {
            return entry.codepoint < value;
        });
    return found != end && found->codepoint == codepoint ? found : nullptr;
}

void canonical_decompose(std::uint32_t codepoint, std::vector<std::uint32_t> & output) {
    constexpr std::uint32_t s_base = 0xAC00;
    constexpr std::uint32_t l_base = 0x1100;
    constexpr std::uint32_t v_base = 0x1161;
    constexpr std::uint32_t t_base = 0x11A7;
    constexpr std::uint32_t l_count = 19;
    constexpr std::uint32_t v_count = 21;
    constexpr std::uint32_t t_count = 28;
    constexpr std::uint32_t n_count = v_count * t_count;
    constexpr std::uint32_t s_count = l_count * n_count;
    if (codepoint >= s_base && codepoint < s_base + s_count) {
        const auto index = codepoint - s_base;
        output.push_back(l_base + index / n_count);
        output.push_back(v_base + (index % n_count) / t_count);
        if (index % t_count != 0) output.push_back(t_base + index % t_count);
        return;
    }
    const auto * decomposition = find_decomposition(codepoint);
    if (!decomposition) {
        output.push_back(codepoint);
        return;
    }
    for (std::size_t index = 0; index < decomposition->length; ++index) {
        canonical_decompose(
            unicode::decomposition_data[decomposition->offset + index], output);
    }
}

std::uint32_t compose_pair(std::uint32_t first, std::uint32_t second) {
    constexpr std::uint32_t s_base = 0xAC00;
    constexpr std::uint32_t l_base = 0x1100;
    constexpr std::uint32_t v_base = 0x1161;
    constexpr std::uint32_t t_base = 0x11A7;
    constexpr std::uint32_t l_count = 19;
    constexpr std::uint32_t v_count = 21;
    constexpr std::uint32_t t_count = 28;
    constexpr std::uint32_t n_count = v_count * t_count;
    constexpr std::uint32_t s_count = l_count * n_count;
    if (first >= l_base && first < l_base + l_count &&
        second >= v_base && second < v_base + v_count) {
        return s_base + ((first - l_base) * v_count + second - v_base) * t_count;
    }
    if (first >= s_base && first < s_base + s_count &&
        (first - s_base) % t_count == 0 &&
        second > t_base && second < t_base + t_count) {
        return first + second - t_base;
    }
    const auto begin = std::begin(unicode::compositions);
    const auto end = std::end(unicode::compositions);
    const auto found = std::lower_bound(
        begin,
        end,
        std::pair<std::uint32_t, std::uint32_t>{first, second},
        [](const unicode::Composition & entry, const auto & value) {
            return entry.first < value.first ||
                (entry.first == value.first && entry.second < value.second);
        });
    return found != end && found->first == first && found->second == second
        ? found->result
        : 0;
}

std::vector<std::uint32_t> normalize_nfc(const std::vector<std::uint32_t> & input) {
    std::vector<std::uint32_t> decomposed;
    decomposed.reserve(input.size());
    for (const auto codepoint : input) canonical_decompose(codepoint, decomposed);
    for (std::size_t index = 1; index < decomposed.size(); ++index) {
        const auto current_class = combining_class(decomposed[index]);
        if (current_class == 0) continue;
        std::size_t position = index;
        while (position > 0) {
            const auto previous_class = combining_class(decomposed[position - 1]);
            if (previous_class == 0 || previous_class <= current_class) break;
            std::swap(decomposed[position], decomposed[position - 1]);
            --position;
        }
    }
    if (decomposed.empty()) return {};

    std::vector<std::uint32_t> composed;
    composed.reserve(decomposed.size());
    composed.push_back(decomposed.front());
    std::size_t starter_position = 0;
    std::uint32_t starter = decomposed.front();
    std::uint8_t last_class = 0;
    for (std::size_t index = 1; index < decomposed.size(); ++index) {
        const auto codepoint = decomposed[index];
        const auto current_class = combining_class(codepoint);
        const auto replacement = compose_pair(starter, codepoint);
        if (replacement != 0 && (last_class == 0 || last_class < current_class)) {
            composed[starter_position] = replacement;
            starter = replacement;
            continue;
        }
        if (current_class == 0) {
            starter_position = composed.size();
            starter = codepoint;
        }
        composed.push_back(codepoint);
        last_class = current_class;
    }
    return composed;
}

std::string encode_codepoints(
    const std::vector<std::uint32_t> & values,
    std::size_t begin,
    std::size_t end) {
    std::string output;
    for (std::size_t index = begin; index < end; ++index) append_utf8(output, values[index]);
    return output;
}

std::size_t contraction_length(
    const std::vector<std::uint32_t> & text,
    std::size_t offset) {
    if (text[offset] != '\'' || offset + 1 >= text.size()) return 0;
    static const std::array<const char *, 7> endings = {"s", "t", "re", "ve", "m", "ll", "d"};
    for (const char * ending : endings) {
        std::size_t length = 0;
        while (ending[length] != '\0') ++length;
        if (offset + 1 + length > text.size()) continue;
        bool matches = true;
        for (std::size_t index = 0; index < length; ++index) {
            auto codepoint = text[offset + 1 + index];
            if (codepoint >= 'A' && codepoint <= 'Z') codepoint += 'a' - 'A';
            if (codepoint == 0x17F) codepoint = 's'; // Unicode case-insensitive regex: long s
            if (codepoint != static_cast<unsigned char>(ending[index])) {
                matches = false;
                break;
            }
        }
        if (matches) return length + 1;
    }
    return 0;
}

std::vector<std::string> pieces(const std::vector<std::uint32_t> & text, bool three_digits) {
    std::vector<std::string> pieces;
    for (std::size_t offset = 0; offset < text.size();) {
        const auto contraction = contraction_length(text, offset);
        if (contraction != 0) {
            pieces.push_back(encode_codepoints(text, offset, offset + contraction));
            offset += contraction;
            continue;
        }

        std::size_t end = offset;
        if (is_letter(text[offset])) {
            end = offset + 1;
        } else if (!is_newline(text[offset]) && !is_letter(text[offset]) &&
                   !is_number(text[offset]) && offset + 1 < text.size() &&
                   is_letter(text[offset + 1])) {
            end = offset + 2;
        }
        if (end != offset) {
            while (end < text.size() && is_letter(text[end])) ++end;
            pieces.push_back(encode_codepoints(text, offset, end));
            offset = end;
            continue;
        }

        if (is_number(text[offset])) {
            end=offset+1;
            while(three_digits&&end<text.size()&&end<offset+3&&is_number(text[end]))++end;
            pieces.push_back(encode_codepoints(text,offset,end));offset=end;
            continue;
        }

        end = offset;
        if (text[end] == 0x20 && end + 1 < text.size() &&
            !is_whitespace(text[end + 1]) && !is_letter(text[end + 1]) &&
            !is_number(text[end + 1])) {
            ++end;
        }
        const auto symbols_begin = end;
        while (end < text.size() && !is_whitespace(text[end]) &&
               !is_letter(text[end]) && !is_number(text[end])) {
            ++end;
        }
        if (end > symbols_begin) {
            while (end < text.size() && is_newline(text[end])) ++end;
            pieces.push_back(encode_codepoints(text, offset, end));
            offset = end;
            continue;
        }

        if (is_whitespace(text[offset])) {
            end = offset;
            std::size_t last_newline_end = offset;
            while (end < text.size() && is_whitespace(text[end])) {
                ++end;
                if (is_newline(text[end - 1])) last_newline_end = end;
            }
            if (last_newline_end > offset) {
                pieces.push_back(encode_codepoints(text, offset, last_newline_end));
                offset = last_newline_end;
                continue;
            }
            if (end == text.size()) {
                pieces.push_back(encode_codepoints(text, offset, end));
                offset = end;
                continue;
            }
            if (end - offset > 1) {
                pieces.push_back(encode_codepoints(text, offset, end - 1));
                offset = end - 1;
                continue;
            }
            pieces.push_back(encode_codepoints(text, offset, end));
            offset = end;
            continue;
        }
        throw std::logic_error("MuLaCover Qwen2 pre-tokenizer did not consume input");
    }
    return pieces;
}


}
