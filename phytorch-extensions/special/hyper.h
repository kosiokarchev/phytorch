#include "hyperhead.h"
#include "gammahead.h"


COMPLEX_TEMPLATE
auto nint_distance(T x) {
    scalar_t n = round(x.real());
    return std::make_tuple((int) n, x != n ? log2(abs(x-n)) : -TINF);
}


COMPLEX_TEMPLATE T hypsum(const vector<T>& a_s, const vector<T>& b_s, T z, HYP_KWARGS_MAXTERMS) {
    std::ostringstream oss;

    T s = 1, t = 1;
    for (auto k=0; k<maxterms; ++k){
        for (const auto& a: a_s) t *= (a+k);
        for (const auto& b: b_s) t /= (b+k);
        t /= k+1; t *= z; s += t;
        if (abs(t) < TEPS) return s;
    }
    return TNAN;
    // throw no_convergence(maxterms);
}


COMPLEX_TEMPLATE T hyper(const vector<T>& a_s, const vector<T>& b_s, T z, HYP_KWARGS) {
    auto p = a_s.size(), q = b_s.size();

    // TODO: Reduce degree by eliminating common parameters

    if (p == 0) {
        if (q == 0) return exp(z);
        if (q == 1) return hyp0f1_<scalar_t>(b_s[0], z, force_series, maxterms);
    }
    // } if (p == 1) {
    //     if (q == 0) return hyp1f0<scalar_t>(a_s[0], z, force_series, maxterms);
    //     if (q == 1) return hyp1f1<scalar_t>(a_s[0], b_s[0], z, force_series, maxterms);
    //     if (q == 2) return hyp1f2<scalar_t>(a_s[0], b_s[0], b_s[1], z, force_series, maxterms);
    // }  if (p == 2) {
    //     if (q == 0) return hyp2f0<scalar_t>(a_s[0], a_s[1], z, force_series, maxterms);
    //     if (q == 1) return hyp2f1_<scalar_t>(a_s[0], a_s[1], b_s[0], z, force_series, maxterms);
    //     if (q == 2) return hyp2f2<scalar_t>(a_s[0], a_s[1], b_s[0], b_s[1], z, force_series, maxterms);
    //     if (q == 3) return hyp2f3<scalar_t>(a_s[0], a_s[1], b_s[0], b_s[1], b_s[2], z, force_series, maxterms);
    // }
    // if (p == q+1) return hypq1fq(p, q, a_s, b_s, z);
    // if (p > q+1 and not force_series) return hyp_borel(p, q, a_s, b_s, z);

    return hypsum<scalar_t>(a_s, b_s, z, maxterms);
}


COMPLEX_TEMPLATE auto hyp0f1_series(T b, T z) {
    auto jw = sqrt(-z) * T(0, 1);
    auto u = 1 / (4 * jw);
    auto c = ltrl(0.5) - b;
    auto E = exp(2 * jw);

    return series_return_t<T>{
        {{-jw, E}, {c, -1}, {}, {}, {b-ltrl(0.5), ltrl(1.5)-b}, {}, -u},
        {{jw, E}, {c, 1}, {}, {}, {b-ltrl(0.5), ltrl(1.5)-b}, {}, u}
    };
}

// INSTANTIATE_COMPLEX_TEMPLATE_NOARGS(hyp0f1_, float)(complex<float>, complex<float>, HYP_KWARGS_TYPES);
// INSTANTIATE_COMPLEX_TEMPLATE_NOARGS(hyp0f1_, double)(complex<double>, complex<double>, HYP_KWARGS_TYPES);
COMPLEX_TEMPLATE T hyp0f1_(T b, T z, HYP_KWARGS) {
    if (abs(z) > 128 and not force_series) {
        // try {
            return ltrl(R2SQRTPI) * gamma<scalar_t>(b) * hypercomb<scalar_t>(
                (series_t<T>) ([b, z] (vector<T> params) {return hyp0f1_series<scalar_t>(b, z);}),
                vector<T>{}, true, maxterms);
        // } catch (no_convergence& exc) { if (force_series) throw exc; }
    }
    return hypsum<scalar_t>(vector<T>{}, vector<T>{b}, z, maxterms);
}


COMPLEX_TEMPLATE T hyp1f0_(T a, T z, HYP_KWARGS) { return pow(1-z, -a); }

COMPLEX_TEMPLATE auto hyp1f1_series(T z, T a, T b) {
    auto E = exp(T(0, M_PI) * (z.imag()<0 ? -a : a));
    return series_return_t<T>{
        {{E, z}, {1, -a}, {b}, {b-a}, {a, 1+a-b}, {}, -1/z},
        {{exp(z), z}, {1, a-b}, {b}, {a}, {b-a, 1-a}, {}, 1/z}
    };
}

COMPLEX_TEMPLATE T hyp1f1_(T a, T b, T z, HYP_KWARGS) {
    if (not z) return 1;
    if (abs(z) > pow(ltrl(2), ltrl(6)) and not is_nonpositive_int(a)) {
        // if (isinf(z)) {
        //     if (abs(a)/a == abs(b)/b == abs(z)/z == 1) return TINF;
        //     else return TNAN * z;
        // }
        return hypercomb<scalar_t>(
            (series_t<T>) ([z] (vector<T> params) {return hyp1f1_series<scalar_t>(z, params[0], params[1]);}),
            vector<T>{a, b}, true, maxterms
        );
    }

    return hypsum<scalar_t>(vector<T>{a}, vector<T>{b}, z, maxterms);
}


COMPLEX_TEMPLATE
auto hypercomb_check_need_perturb(const series_return_t<T>& terms) {
    auto perturb = false;
    vector<size_t> discard;

    for (auto term_index=0; term_index<terms.size(); ++term_index) {
        auto& [w_s, c_s, alpha_s, beta_s, a_s, b_s, z] = terms[term_index];
        auto have_singular_nongamma_weight = false;

        // Avoid division by zero in leading factors
        // TODO: near divisions by zero
        for (auto w=w_s.begin(), c=c_s.begin(); w < w_s.end(); ++w, ++c)
            if (abs(*w) < 2*TEPS and (*c).real() <= 0 and (*c) != ltrl(0)) {
                perturb = have_singular_nongamma_weight = true;
                break;
            }

        // Check for gamma and series poles and near-poles
        size_t pole_count[]{0, 0, 0};
        for (auto data_index=0; data_index < 3; ++data_index) {
            auto& data = data_index == 0 ? alpha_s : data_index == 1 ? beta_s : b_s;
            for (auto x=data.begin(); x < data.end(); ++x) {
                auto [n, d] = nint_distance<scalar_t>(*x);
                if (n > 0) continue;
                if (d == -TINF) {
                    // OK if we have a polynomial
                    auto ok = false;
                    if (data_index == 2)
                        for (const auto& u: a_s)
                            if (u and u.real() > n) {ok = true; break;}
                    if (ok) continue;
                    ++pole_count[data_index];
                }
            }
        }
        if (pole_count[1] > pole_count[0] + pole_count[2] and not have_singular_nongamma_weight)
            discard.push_back(term_index);
        else if (pole_count[0] or pole_count[1] or pole_count[2])
            perturb = true;
    }

    return std::make_tuple(perturb, discard);
}


INSTANTIATE_COMPLEX_TEMPLATE_NOARGS(hypercomb, float)(series_t<complex<float>>, const vector<complex<float>>&, HYP_KWARGS);
INSTANTIATE_COMPLEX_TEMPLATE_NOARGS(hypercomb, double)(series_t<complex<double>>, const vector<complex<double>>&, HYP_KWARGS);
COMPLEX_TEMPLATE T hypercomb(series_t<T> function, const vector<T>& params, HYP_KWARGS) {
    auto terms = function(params);
    auto [perturb, discard] = hypercomb_check_need_perturb<scalar_t>(terms);

    if (perturb) {
        vector<T> params_(params.size());
        scalar_t h = pow(ltrl(2), -(floor(ltrl(0.3) * TPREC)));

        for (auto k=0; k<params.size(); ++k) {
            params_[k] = params[k] + h;
            // Heuristically ensure that the perturbations are "independent" so that two perturbations
            // don't accidentally cancel each other out in a subtraction.
            h += h / (k+1);
        }
        terms = function(params_);
    }

    T ret = 0;
    for (size_t i=0; i < terms.size(); ++i)
        if (find(discard.begin(), discard.end(), i) == discard.end()) {
            auto [w_s, c_s, alpha_s, beta_s, a_s, b_s, z] = terms[i];
            auto v = hyper<scalar_t>(a_s, b_s, z, force_series, maxterms);
            for (const auto& a: alpha_s) v *= gamma<scalar_t>(a);
            for (const auto& b: beta_s)  v /= gamma<scalar_t>(b);
            for (auto w=w_s.begin(), c=c_s.begin(); w<w_s.end(); ++w, ++c) v *= pow(*w, *c);
            ret += v;
        }
    return ret;
}