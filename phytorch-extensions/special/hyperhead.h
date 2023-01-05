#pragma once

#define HYP_KWARGS_FORCE_TYPE const bool&
#define HYP_KWARGS_FORCE_NAME force_series
#define HYP_KWARGS_FORCE_VALUE false
#define HYP_KWARGS_FORCE HYP_KWARGS_FORCE_TYPE HYP_KWARGS_FORCE_NAME
#define HYP_KWARGS_FORCE_H HYP_KWARGS_FORCE=HYP_KWARGS_FORCE_VALUE
#define HYP_KWARGS_MAXTERMS_TYPE const size_t&
#define HYP_KWARGS_MAXTERMS_NAME maxterms
#define HYP_KWARGS_MAXTERMS_VALUE 6000
#define HYP_KWARGS_MAXTERMS HYP_KWARGS_MAXTERMS_TYPE HYP_KWARGS_MAXTERMS_NAME
#define HYP_KWARGS_MAXTERMS_H HYP_KWARGS_MAXTERMS=HYP_KWARGS_MAXTERMS_VALUE
#define HYP_KWARGS_TYPES HYP_KWARGS_FORCE_TYPE, HYP_KWARGS_MAXTERMS_TYPE
#define HYP_KWARGS_NAMES HYP_KWARGS_FORCE_NAME, HYP_KWARGS_MAXTERMS_NAME
#define HYP_KWARGS HYP_KWARGS_FORCE, HYP_KWARGS_MAXTERMS
#define HYP_KWARGS_H HYP_KWARGS_FORCE_H, HYP_KWARGS_MAXTERMS_H

#include <string>
#include <exception>
#include <functional>
#include <sstream>
#include "../common/templating.h"


// #ifdef __CUDACC__
// #include <thrust/device_vector.h>
// #define vector thrust::device_vector
// #include <thrust/tuple.h>
// #define tuple thrust::tuple
// #else
#include <vector>
using std::vector;
#include <tuple>
using std::tuple;
// #endif

// #include <cuda/std/array>
// using cuda::std::vector;

template <typename T>
using term_t = tuple<vector<T>, vector<T>, vector<T>, vector<T>, vector<T>, vector<T>, T>;

template <typename T>
using series_return_t = vector<term_t<T>>;

template <typename T>
using series_t = std::function<series_return_t<T>(vector<T>)>;

class no_convergence: public std::runtime_error {
private:
    static std::string init(size_t maxterms) {
        std::ostringstream os;
        os << "hypsum failed to converge in " << maxterms << " terms";
        return os.str();
    }
public:
    explicit no_convergence(size_t maxterms): std::runtime_error(init(maxterms)) {}
};


COMPLEX_TEMPLATE_H T hypsum(const vector<T>& a_s, const vector<T>& b_s, T z, HYP_KWARGS_MAXTERMS_H);

COMPLEX_TEMPLATE_H T hyper(const vector<T>& a_s, const vector<T>& b_s, T z, HYP_KWARGS_H);
COMPLEX_TEMPLATE_H T hyp0f1(T b, T z, HYP_KWARGS_H);
COMPLEX_TEMPLATE_H T hyp1f0(T a, T z, HYP_KWARGS_H);
COMPLEX_TEMPLATE_H T hyp1f1(T a, T b, T z, HYP_KWARGS_H);
COMPLEX_TEMPLATE_H T hyp1f2(T a, T b1, T b2, T z, HYP_KWARGS_H);
COMPLEX_TEMPLATE_H T hyp2f0(T a1, T a2, T z, HYP_KWARGS_H);
COMPLEX_TEMPLATE_H T hyp2f1_(T a1, T a2, T b, T z, HYP_KWARGS_H);
COMPLEX_TEMPLATE_H T hyp2f2(T a1, T a2, T b1, T b2, T z, HYP_KWARGS_H);
COMPLEX_TEMPLATE_H T hyp2f3(T a1, T a2, T b1, T b2, T b3, T z, HYP_KWARGS_H);

COMPLEX_TEMPLATE_H T hypercomb(series_t<T> function, const vector<T>& params, HYP_KWARGS_H);

COMPLEX_TEMPLATE_H T meijerg(size_t m, size_t n, size_t p, size_t q, const vector<T>& args, T z, T r, HYP_KWARGS);
COMPLEX_TEMPLATE_H T Tgamma(size_t m, T a, T z);
