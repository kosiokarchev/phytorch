#pragma once

#include <tuple>
#include <array>

template <typename T, size_t n, size_t... Is>
auto array_to_tuple(std::array<T, n> const& arr, std::index_sequence<Is...>) {
    return std::make_tuple(arr[Is]...);
}
