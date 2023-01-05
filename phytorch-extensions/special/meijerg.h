#include "hyperhead.h"


COMPLEX_TEMPLATE auto meijerg_series1(size_t m, size_t n, size_t p, size_t q, T z, T r, vector<T> args) {
    auto a = args.begin(), b = args.begin() + p;
    series_return_t<T> ret(m);
    for (auto k=0; k<m; ++k) {
        auto& term = ret[k];
        std::get<0>(term) = {z};
        std::get<1>(term) = {b[k] / r};
        for (auto j=0; j<m; ++j)
            if (j != k)
                std::get<2>(term).push_back(b[j] - b[k]);
        for (auto j=0; j<n; ++j)
            std::get<2>(term).push_back(1 - a[j] + b[k]);
        for (auto j=n; j<p; ++j)
            std::get<3>(term).push_back(a[j] - b[k]);
        for (auto j=m; j<q; ++j)
            std::get<3>(term).push_back(1 - b[j] + b[k]);
        for (auto j=0; j<p; ++j)
            std::get<4>(term).push_back(1 - a[j] + b[k]);
        for (auto j=0; j<q; ++j)
            if (j != k)
                std::get<5>(term).push_back(1 - b[j] + b[k]);
        std::get<6>(term) = pow(ltrl(-1), ltrl(p)-ltrl(m)-ltrl(n)) * pow(z, 1/r);
    }
    return ret;
}

COMPLEX_TEMPLATE series_return_t<T> meijerg_series2(size_t m, size_t n, size_t p, size_t q, T z, T r, vector<T> args) {
    printf("series 2\n");
    auto a = args.begin(), b = args.begin() + p;
    series_return_t<T> ret(n);
    for (auto k=0; k<n; ++k) {
        auto& term = ret[k];
        std::get<0>(term) = {z};
        std::get<1>(term) = {(a[k] - 1) / r};
        for (auto j=0; j<n; ++j)
            if (j != k)
                std::get<2>(term).push_back(a[k] - a[j]);
        for (auto j=0; j<m; ++j)
            std::get<2>(term).push_back(1 - a[k] + b[j]);
        for (auto j=m; j<q; ++j)
            std::get<3>(term).push_back(a[k] - b[j]);
        for (auto j=n; j<p; ++j)
            std::get<3>(term).push_back(1 - a[k] + a[j]);
        for (auto j=0; j<q; ++j)
            std::get<4>(term).push_back(1 - a[k] + b[j]);
        for (auto j=0; j<p; ++j)
            if (j != k)
                std::get<5>(term).push_back(1 + a[j] - a[k]);
        std::get<6>(term) = pow(ltrl(-1), ltrl(q-m-n)) / pow(z, 1/r);
    }
    return ret;
}


INSTANTIATE_COMPLEX_TEMPLATE_NOARGS(meijerg, float)(size_t, size_t, size_t, size_t, const vector<complex<float>>&, complex<float>, complex<float>, HYP_KWARGS_TYPES);
INSTANTIATE_COMPLEX_TEMPLATE_NOARGS(meijerg, double)(size_t, size_t, size_t, size_t, const vector<complex<double>>&, complex<double>, complex<double>, HYP_KWARGS_TYPES);
COMPLEX_TEMPLATE T meijerg(size_t m, size_t n, size_t p, size_t q, const vector<T>& args, T z, T r, HYP_KWARGS) {
    auto series =
        p < q ? meijerg_series1<scalar_t> :
        p > q ? meijerg_series2<scalar_t> :
        m + n == p and abs(z) > 1 ? meijerg_series2<scalar_t> : meijerg_series1<scalar_t>;
    return hypercomb<scalar_t>(
            (series_t<T>) ([&](vector<T> _args) {return series(m, n, p, q, z, r, _args);}),
            args, HYP_KWARGS_NAMES);
}


INSTANTIATE_COMPLEX_TEMPLATE_NOARGS(Tgamma, float)(size_t, complex<float>, complex<float>);
INSTANTIATE_COMPLEX_TEMPLATE_NOARGS(Tgamma, double)(size_t, complex<double>, complex<double>);
COMPLEX_TEMPLATE T Tgamma(size_t m, T a, T z)  {
    vector<T> args(m+m-1);
    args[m-1] = a - 1;
    for (auto i=m; i<m+m-1; ++i)
        args[i] = -1;
    return meijerg<scalar_t>(m, 0, m-1, m, args, z, T(1), HYP_KWARGS_FORCE_VALUE, HYP_KWARGS_MAXTERMS_VALUE);
}
