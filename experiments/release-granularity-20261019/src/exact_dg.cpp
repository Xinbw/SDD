// Canonne--Kamath--Steinke discrete Gaussian rejection sampler.
// All Bernoulli decisions use exact rational probabilities and uniform bits.
// Output is post-processed modulo 2^64, matching the aggregation ring.
#include <openssl/evp.h>
#include <openssl/sha.h>
#include <boost/multiprecision/cpp_int.hpp>
#include <omp.h>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

using u128 = unsigned __int128;
using boost::multiprecision::cpp_int;

class Bits {
    EVP_CIPHER_CTX* ctx;
    unsigned char buf[8192];
    size_t pos = sizeof(buf);
public:
    Bits(const unsigned char* master, uint64_t stream) {
        unsigned char input[40], key[32], iv[16] = {};
        memcpy(input, master, 32); memcpy(input+32, &stream, 8);
        SHA256(input, sizeof(input), key);
        ctx = EVP_CIPHER_CTX_new();
        if (!ctx || !EVP_EncryptInit_ex(ctx, EVP_chacha20(), nullptr, key, iv))
            throw std::runtime_error("ChaCha20 initialization failed");
    }
    ~Bits() { EVP_CIPHER_CTX_free(ctx); }
    uint64_t next() {
        if (pos == sizeof(buf)) {
            static const unsigned char zeros[8192] = {};
            int length = 0;
            if (!EVP_EncryptUpdate(ctx, buf, &length, zeros, sizeof(buf)) || length != sizeof(buf))
                throw std::runtime_error("ChaCha20 stream failed");
            pos = 0;
        }
        uint64_t value; memcpy(&value, buf+pos, 8); pos += 8; return value;
    }
    uint64_t uniform(uint64_t bound) {
        if (bound == 1) return 0;
        unsigned bits = 64-__builtin_clzll(bound-1);
        uint64_t mask = bits == 64 ? UINT64_MAX : (uint64_t(1)<<bits)-1;
        uint64_t value; do { value = next() & mask; } while (value >= bound);
        return value;
    }
    u128 uniform128(u128 bound) {
        if (bound <= UINT64_MAX) return uniform(uint64_t(bound));
        u128 top = bound-1;
        unsigned bits = 64 + 64-__builtin_clzll(uint64_t(top>>64));
        u128 mask = bits == 128 ? ~u128(0) : (u128(1)<<bits)-1;
        u128 value;
        do { value = ((u128(next())<<64) | next()) & mask; } while (value >= bound);
        return value;
    }
    cpp_int uniform_big(const cpp_int& bound) {
        unsigned bits = boost::multiprecision::msb(bound-1)+1;
        cpp_int mask = (cpp_int(1)<<bits)-1, value;
        do {
            value = 0;
            for (unsigned j=0; j<(bits+63)/64; ++j) { value <<= 64; value |= next(); }
            value &= mask;
        } while (value >= bound);
        return value;
    }
};

bool exp_fraction(Bits& bits, uint64_t numerator, uint64_t denominator) {
    if (!numerator) return true;
    uint64_t k = 1;
    while (bits.uniform128(u128(denominator)*k) < numerator) {
        if (k == UINT64_MAX) {
            cpp_int big_k = cpp_int(k)+1;
            while (bits.uniform_big(cpp_int(denominator)*big_k) < numerator) ++big_k;
            return (big_k & 1) != 0;
        }
        ++k;
    }
    return (k & 1) != 0;
}

bool exp_rational(Bits& bits, u128 numerator, uint64_t denominator) {
    u128 whole = numerator/denominator;
    while (whole) {
        if (!exp_fraction(bits, 1, 1)) return false;
        --whole;
    }
    return exp_fraction(bits, uint64_t(numerator%denominator), denominator);
}

bool exp_big(Bits& bits, const cpp_int& numerator, uint64_t denominator) {
    cpp_int whole = numerator/denominator;
    while (whole > 0) {
        if (!exp_fraction(bits, 1, 1)) return false;
        --whole;
    }
    return exp_fraction(bits, (numerator%denominator).convert_to<uint64_t>(), denominator);
}

int64_t discrete_gaussian(Bits& bits, uint64_t sigma) {
    const uint64_t denominator = 2*sigma*sigma;
    while (true) {
        uint64_t u;
        do { u = bits.uniform(sigma); } while (!exp_fraction(bits, u, sigma));
        uint64_t g = 0;
        bool large = false;
        cpp_int big_g;
        while (exp_fraction(bits, 1, 1)) {
            if (g == UINT64_MAX) { large = true; big_g = cpp_int(g)+1; break; }
            ++g;
        }
        if (large) {
            while (exp_fraction(bits, 1, 1)) ++big_g;
        }
        bool negative = bits.next() & 1;
        u128 y = u128(u)+u128(sigma)*g;
        if (!large && !y && negative) continue;
        uint64_t output;
        if (large || y > UINT64_MAX) {
            cpp_int big_y = cpp_int(u)+cpp_int(sigma)*(large ? big_g : cpp_int(g));
            cpp_int difference = big_y-sigma;
            if (!exp_big(bits, difference*difference, denominator)) continue;
            output = (big_y & ((cpp_int(1)<<64)-1)).convert_to<uint64_t>();
        } else {
            u128 difference = y >= sigma ? y-sigma : sigma-y;
            if (!exp_rational(bits, difference*difference, denominator)) continue;
            output = uint64_t(y);
        }
        if (negative) output = uint64_t(0)-output;
        int64_t signed_output; memcpy(&signed_output, &output, 8); return signed_output;
    }
}

extern "C" int dg_fill(int64_t* output, uint64_t count, uint64_t sigma,
                       const unsigned char* key, int threads) {
    if (!sigma || sigma > 1000000000 || threads < 1 || !output || !key) return 1;
    #pragma omp parallel num_threads(threads)
    {
        int tid = omp_get_thread_num(), total = omp_get_num_threads();
        uint64_t begin = count*tid/total, end = count*(tid+1)/total;
        Bits bits(key, uint64_t(tid));
        for (uint64_t i=begin; i<end; ++i) output[i] = discrete_gaussian(bits, sigma);
    }
    return 0;
}
