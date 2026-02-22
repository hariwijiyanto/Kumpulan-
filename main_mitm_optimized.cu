// General Arithmetic Optimizations for secp256k1:
// Modular arithmetic operations, especially multiplication and inverse, are critical for ECC performance.
// For modular multiplication (mul_mod_device), this implementation uses a specialized reduction method
// tailored for the secp256k1 prime modulus (p = 2^256 - 2^32 - 977). While efficient for this specific prime,
// for general prime moduli, Montgomery multiplication is a commonly used and highly optimized alternative.
// For modular inverse (mod_inverse), Fermat's Little Theorem is used here (a^(p-2) mod p).
// For a fixed modulus like const_p, the Extended Euclidean Algorithm can be a more efficient approach
// for calculating modular inverses, often involving fewer and simpler operations for certain implementations.

#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

#define BIGINT_WORDS 8

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

struct BigInt {
    uint32_t data[BIGINT_WORDS];
};

struct ECPoint {
    BigInt x, y;
    bool infinity;
};

struct ECPointJac {
    BigInt X, Y, Z;
    bool infinity;
};

struct CompressedPubKey {
    uint8_t data[33];
};

struct DPEntry {
    uint64_t fp;
    uint64_t k_trap;
};

__constant__ BigInt const_p;
__constant__ ECPointJac const_G_jacobian;
__constant__ BigInt const_n;
__device__ ECPointJac const_G_table[256];
__constant__ ECPointJac const_target_jacobian;

// Fungsi utilitas BigInt
__host__ __device__ __forceinline__ void init_bigint(BigInt *x, uint32_t val) {
    for (int i = 0; i < BIGINT_WORDS; i++) {
        x->data[i] = 0;
    }
    x->data[0] = val;
}

__host__ __device__ __forceinline__ void copy_bigint(BigInt *dest, const BigInt *src) {
    for (int i = 0; i < BIGINT_WORDS; i++) {
        dest->data[i] = src->data[i];
    }
}

__host__ __device__ __forceinline__ int compare_bigint(const BigInt *a, const BigInt *b) {
    for (int i = BIGINT_WORDS - 1; i >= 0; i--) {
        if (a->data[i] > b->data[i]) return 1;
        if (a->data[i] < b->data[i]) return -1;
    }
    return 0;
}

__host__ __device__ __forceinline__ bool is_zero(const BigInt *a) {
    for (int i = 0; i < BIGINT_WORDS; i++) {
        if (a->data[i]) return false;
    }
    return true;
}

__host__ __device__ __forceinline__ int get_bit(const BigInt *a, int i) {
    int word_idx = i >> 5;
    int bit_idx = i & 31;
    if (word_idx >= BIGINT_WORDS) return 0;
    return (a->data[word_idx] >> bit_idx) & 1;
}

__host__ __device__ __forceinline__ void ptx_u256Add(BigInt *res, const BigInt *a, const BigInt *b) {
    asm volatile (
        "add.cc.u32 %0, %8, %16; \n\t"
        "addc.cc.u32 %1, %9, %17; \n\t"
        "addc.cc.u32 %2, %10, %18; \n\t"
        "addc.cc.u32 %3, %11, %19; \n\t"
        "addc.cc.u32 %4, %12, %20; \n\t"
        "addc.cc.u32 %5, %13, %21; \n\t"
        "addc.cc.u32 %6, %14, %22; \n\t"
        "addc.u32    %7, %15, %23; \n\t"
        : "=r"(res->data[0]), "=r"(res->data[1]), "=r"(res->data[2]), "=r"(res->data[3]),
          "=r"(res->data[4]), "=r"(res->data[5]), "=r"(res->data[6]), "=r"(res->data[7])
        : "r"(a->data[0]), "r"(a->data[1]), "r"(a->data[2]), "r"(a->data[3]),
          "r"(a->data[4]), "r"(a->data[5]), "r"(a->data[6]), "r"(a->data[7]),
          "r"(b->data[0]), "r"(b->data[1]), "r"(b->data[2]), "r"(b->data[3]),
          "r"(b->data[4]), "r"(b->data[5]), "r"(b->data[6]), "r"(b->data[7])
    );
}

__host__ __device__ __forceinline__ void ptx_u256Sub(BigInt *res, const BigInt *a, const BigInt *b) {
    asm volatile (
        "sub.cc.u32 %0, %8, %16; \n\t"
        "subc.cc.u32 %1, %9, %17; \n\t"
        "subc.cc.u32 %2, %10, %18; \n\t"
        "subc.cc.u32 %3, %11, %19; \n\t"
        "subc.cc.u32 %4, %12, %20; \n\t"
        "subc.cc.u32 %5, %13, %21; \n\t"
        "subc.cc.u32 %6, %14, %22; \n\t"
        "subc.u32    %7, %15, %23; \n\t"
        : "=r"(res->data[0]), "=r"(res->data[1]), "=r"(res->data[2]), "=r"(res->data[3]),
          "=r"(res->data[4]), "=r"(res->data[5]), "=r"(res->data[6]), "=r"(res->data[7])
        : "r"(a->data[0]), "r"(a->data[1]), "r"(a->data[2]), "r"(a->data[3]),
          "r"(a->data[4]), "r"(a->data[5]), "r"(a->data[6]), "r"(a->data[7]),
          "r"(b->data[0]), "r"(b->data[1]), "r"(b->data[2]), "r"(b->data[3]),
          "r"(b->data[4]), "r"(b->data[5]), "r"(b->data[6]), "r"(b->data[7])
    );
}

__host__ __device__ __forceinline__ void bigint_add_bigint(BigInt *res, const BigInt *a, const BigInt *b) {
    // Optimization Note: Consider replacing this generic software-based addition with PTX intrinsics or CUDA built-in functions (e.g., __builtin_addc) for optimized carry-chain handling, similar to `ptx_u256Add` for potentially faster BigInt addition.

    uint64_t carry = 0;
    for (int i = 0; i < BIGINT_WORDS; i++) {
        uint64_t sum = (uint64_t)a->data[i] + (uint64_t)b->data[i] + carry;
        res->data[i] = (uint32_t)sum;
        carry = sum >> 32;
    }
}

__host__ __device__ __forceinline__ void bigint_add_uint64(BigInt *res, const BigInt *a, uint64_t b) {
    uint64_t carry = b;
    for (int i = 0; i < BIGINT_WORDS; i++) {
        uint64_t sum = (uint64_t)a->data[i] + carry;
        res->data[i] = (uint32_t)sum;
        carry = sum >> 32;
        if (carry == 0) {
            for (int j = i + 1; j < BIGINT_WORDS; j++) {
                res->data[j] = a->data[j];
            }
            break;
        }
    }
}

// PERKALIAN SCALAR SEDERHANA
__device__ __forceinline__ void scalar_mul_simple(BigInt *res, const BigInt *a, const BigInt *b) {
    uint64_t prod[2 * BIGINT_WORDS] = {0};

    for (int i = 0; i < BIGINT_WORDS; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < BIGINT_WORDS; j++) {
            int idx = i + j;
            if (idx < 2 * BIGINT_WORDS) {
                uint64_t product = (uint64_t)a->data[i] * b->data[j];
                uint64_t sum = prod[idx] + product + carry;
                prod[idx] = sum & 0xFFFFFFFF;
                carry = sum >> 32;
            }
        }
        if (i + BIGINT_WORDS < 2 * BIGINT_WORDS) {
            int idx = i + BIGINT_WORDS;
            uint64_t sum = prod[idx] + carry;
            prod[idx] = sum & 0xFFFFFFFF;
        }
    }

    for (int i = 0; i < BIGINT_WORDS; i++) {
        res->data[i] = prod[i];
    }
}

// PERKALIAN SCALAR dengan modulo n
__device__ __forceinline__ void scalar_mul_mod_n(BigInt *res, const BigInt *a, const BigInt *b) {
    BigInt product;
    scalar_mul_simple(&product, a, b);

    if (compare_bigint(&product, &const_n) >= 0) {
        ptx_u256Sub(res, &product, &const_n);
    } else {
        copy_bigint(res, &product);
    }
}

__device__ __forceinline__ void multiply_bigint_by_const(const BigInt *a, uint32_t c, uint32_t result[9]) {
    uint64_t carry = 0;
    for (int i = 0; i < BIGINT_WORDS; i++) {
        uint64_t prod = (uint64_t)a->data[i] * c + carry;
        result[i] = (uint32_t)prod;
        carry = prod >> 32;
    }
    result[8] = (uint32_t)carry;
}

__device__ __forceinline__ void shift_left_word(const BigInt *a, uint32_t result[9]) {
    result[0] = 0;
    for (int i = 0; i < BIGINT_WORDS; i++) {
        result[i+1] = a->data[i];
    }
}

__device__ __forceinline__ void add_9word(uint32_t r[9], const uint32_t addend[9]) {
    uint64_t carry = 0;
    for (int i = 0; i < 9; i++) {
        uint64_t sum = (uint64_t)r[i] + addend[i] + carry;
        r[i] = (uint32_t)sum;
        carry = sum >> 32;
    }
}

__device__ __forceinline__ void convert_9word_to_bigint(const uint32_t r[9], BigInt *res) {
    for (int i = 0; i < BIGINT_WORDS; i++) {
        res->data[i] = r[i];
    }
}

__device__ __forceinline__ void mul_mod_device(BigInt *res, const BigInt *a, const BigInt *b) {
    // Note: This implementation uses a specialized reduction method tailored for secp256k1's prime.
    // For general prime moduli, Montgomery multiplication would be a common alternative for optimization.

    uint32_t prod[2 * BIGINT_WORDS] = {0};
    for (int i = 0; i < BIGINT_WORDS; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < BIGINT_WORDS; j++) {
            uint64_t tmp = (uint64_t)prod[i + j] + (uint64_t)a->data[i] * b->data[j] + carry;
            prod[i + j] = (uint32_t)tmp;
            carry = tmp >> 32;
        }
        prod[i + BIGINT_WORDS] += (uint32_t)carry;
    }

    BigInt L, H;
    for (int i = 0; i < BIGINT_WORDS; i++) {
        L.data[i] = prod[i];
        H.data[i] = prod[i + BIGINT_WORDS];
    }

    uint32_t Rext[9] = {0};
    for (int i = 0; i < BIGINT_WORDS; i++) Rext[i] = L.data[i];
    Rext[8] = 0;

    uint32_t H977[9] = {0};
    multiply_bigint_by_const(&H, 977, H977);
    add_9word(Rext, H977);

    uint32_t Hshift[9] = {0};
    shift_left_word(&H, Hshift);
    add_9word(Rext, Hshift);

    if (Rext[8]) {
        uint32_t extra[9] = {0};
        BigInt extraBI;
        init_bigint(&extraBI, Rext[8]);
        Rext[8] = 0;

        uint32_t extra977[9] = {0}, extraShift[9] = {0};
        multiply_bigint_by_const(&extraBI, 977, extra977);
        shift_left_word(&extraBI, extraShift);

        for (int i = 0; i < 9; i++) extra[i] = extra977[i];
        add_9word(extra, extraShift);
        add_9word(Rext, extra);
    }

    BigInt R_temp;
    convert_9word_to_bigint(Rext, &R_temp);

    if (Rext[8] || compare_bigint(&R_temp, &const_p) >= 0) {
        ptx_u256Sub(&R_temp, &R_temp, &const_p);
    }
    if (compare_bigint(&R_temp, &const_p) >= 0) {
        ptx_u256Sub(&R_temp, &R_temp, &const_p);
    }

    copy_bigint(res, &R_temp);
}

__device__ __forceinline__ void sub_mod_device(BigInt *res, const BigInt *a, const BigInt *b) {
    BigInt temp;
    if (compare_bigint(a, b) < 0) {
         BigInt sum;
         ptx_u256Add(&sum, a, &const_p);
         ptx_u256Sub(&temp, &sum, b);
    } else {
         ptx_u256Sub(&temp, a, b);
    }
    copy_bigint(res, &temp);
}

__device__ __forceinline__ void scalar_mod_n(BigInt *res, const BigInt *a) {
    if (compare_bigint(a, &const_n) >= 0) {
        ptx_u256Sub(res, a, &const_n);
    }
     else {
        copy_bigint(res, a);
    }
}

__device__ __forceinline__ void add_mod_device(BigInt *res, const BigInt *a, const BigInt *b) {
    BigInt sum_ab;
    uint64_t carry = 0;
    for (int i = 0; i < BIGINT_WORDS; ++i) {
         uint64_t word_sum = (uint64_t)a->data[i] + b->data[i] + carry;
         sum_ab.data[i] = (uint32_t)word_sum;
         carry = word_sum >> 32;
    }
    if (carry || compare_bigint(&sum_ab, &const_p) >= 0) {
        ptx_u256Sub(res, &sum_ab, &const_p);
    } else {
        copy_bigint(res, &sum_ab);
    }
}

__device__ void modexp(BigInt *res, const BigInt *base, const BigInt *exp) {
    BigInt result;
    init_bigint(&result, 1);
    BigInt b;
    copy_bigint(&b, base);
    for (int i = 0; i < 256; i++) {
         if (get_bit(exp, i)) {
              mul_mod_device(&result, &result, &b);
         }
         mul_mod_device(&b, &b, &b);
    }
    copy_bigint(res, &result);
}

__device__ void mod_inverse(BigInt *res, const BigInt *a) {
    // Note: For a fixed modulus like const_p, the Extended Euclidean Algorithm could be
    // a more efficient approach than Fermat's Little Theorem for computing modular inverses.

    if (is_zero(a)) {
        init_bigint(res, 0);
        return;
    }
    BigInt p_minus_2, two;
    init_bigint(&two, 2);
    ptx_u256Sub(&p_minus_2, &const_p, &two);
    modexp(res, a, &p_minus_2);
}

__device__ __forceinline__ void point_set_infinity_jac(ECPointJac *P) {
    P->infinity = true;
}

__device__ __forceinline__ void point_copy_jac(ECPointJac *dest, const ECPointJac *src) {
    copy_bigint(&dest->X, &src->X);
    copy_bigint(&dest->Y, &src->Y);
    copy_bigint(&dest->Z, &src->Z);
    dest->infinity = src->infinity;
}

__device__ __forceinline__ void negate_point_jac(ECPointJac *R, const ECPointJac *P) {
    point_copy_jac(R, P);
    sub_mod_device(&R->Y, &const_p, &P->Y);
}

__device__ void double_point_jac(ECPointJac *R, const ECPointJac *P) {
    if (P->infinity || is_zero(&P->Y)) {
        point_set_infinity_jac(R);
        return;
    }
    BigInt A, B, C, D, X3, Y3, Z3, temp, temp2;
    mul_mod_device(&A, &P->Y, &P->Y);
    mul_mod_device(&temp, &P->X, &A);
    init_bigint(&temp2, 4);
    mul_mod_device(&B, &temp, &temp2);
    mul_mod_device(&temp, &A, &A);
    init_bigint(&temp2, 8);
    mul_mod_device(&C, &temp, &temp2);
    mul_mod_device(&temp, &P->X, &P->X);
    init_bigint(&temp2, 3);
    mul_mod_device(&D, &temp, &temp2);
    BigInt D2, two, twoB;
    mul_mod_device(&D2, &D, &D);
    init_bigint(&two, 2);
    mul_mod_device(&twoB, &B, &two);
    sub_mod_device(&X3, &D2, &twoB);
    sub_mod_device(&temp, &B, &X3);
    mul_mod_device(&temp, &D, &temp);
    sub_mod_device(&Y3, &temp, &C);
    init_bigint(&temp, 2);
    mul_mod_device(&temp, &temp, &P->Y);
    mul_mod_device(&Z3, &temp, &P->Z);
    copy_bigint(&R->X, &X3);
    copy_bigint(&R->Y, &Y3);
    copy_bigint(&R->Z, &Z3);
    R->infinity = false;
}

__device__ void add_point_jac(ECPointJac *R, const ECPointJac *P, const ECPointJac *Q) {
    if (P->infinity) { point_copy_jac(R, Q); return; }
    if (Q->infinity) { point_copy_jac(R, P); return; }

    BigInt Z1Z1, Z2Z2, U1, U2, S1, S2, H, R_big, H2, H3, U1H2, X3, Y3, Z3, temp;
    mul_mod_device(&Z1Z1, &P->Z, &P->Z);
    mul_mod_device(&Z2Z2, &Q->Z, &Q->Z);
    mul_mod_device(&U1, &P->X, &Z2Z2);
    mul_mod_device(&U2, &Q->X, &Z1Z1);
    BigInt Z2_cubed, Z1_cubed;
    mul_mod_device(&temp, &Z2Z2, &Q->Z); copy_bigint(&Z2_cubed, &temp);
    mul_mod_device(&temp, &Z1Z1, &P->Z); copy_bigint(&Z1_cubed, &temp);
    mul_mod_device(&S1, &P->Y, &Z2_cubed);
    mul_mod_device(&S2, &Q->Y, &Z1_cubed);

    if (compare_bigint(&U1, &U2) == 0) {
        if (compare_bigint(&S1, &S2) != 0) {
            point_set_infinity_jac(R);
            return;
        } else {
            double_point_jac(R, P);
            return;
        }
    }
    sub_mod_device(&H, &U2, &U1);
    sub_mod_device(&R_big, &S2, &S1);
    mul_mod_device(&H2, &H, &H);
    mul_mod_device(&H3, &H2, &H);
    mul_mod_device(&U1H2, &U1, &H2);
    BigInt R2, two, twoU1H2;
    mul_mod_device(&R2, &R_big, &R_big);
    init_bigint(&two, 2);
    mul_mod_device(&twoU1H2, &U1H2, &two);
    sub_mod_device(&temp, &R2, &H3);
    sub_mod_device(&X3, &temp, &twoU1H2);
    sub_mod_device(&temp, &U1H2, &X3);
    mul_mod_device(&temp, &R_big, &temp);
    mul_mod_device(&Y3, &S1, &H3);
    sub_mod_device(&Y3, &temp, &Y3);
    mul_mod_device(&temp, &P->Z, &Q->Z);
    mul_mod_device(&Z3, &temp, &H);
    copy_bigint(&R->X, &X3);
    copy_bigint(&R->Y, &Y3);
    copy_bigint(&R->Z, &Z3);
    R->infinity = false;
}

__device__ void jacobian_to_affine(ECPoint *R, const ECPointJac *P) {
    if (P->infinity) {
        R->infinity = true;
        init_bigint(&R->x, 0);
        init_bigint(&R->y, 0);
        return;
    }
    BigInt Zinv, Zinv2, Zinv3;
    mod_inverse(&Zinv, &P->Z);
    mul_mod_device(&Zinv2, &Zinv, &Zinv);
    mul_mod_device(&Zinv3, &Zinv2, &Zinv);
    mul_mod_device(&R->x, &P->X, &Zinv2);
    mul_mod_device(&R->y, &P->Y, &Zinv3);
    R->infinity = false;
}

// PRECOMPUTATION FUNCTIONS
__global__ void precompute_G_table_kernel() {
    int idx = threadIdx.x;
    if (idx == 0) {
        ECPointJac current = const_G_jacobian;
        point_copy_jac(&const_G_table[0], &current);

        for (int i = 1; i < 256; i++) {
            double_point_jac(&current, &current);
            point_copy_jac(&const_G_table[i], &current);
        }
    }
}

__device__ void scalar_multiply_jac_precomputed(ECPointJac *result, const BigInt *scalar) {
    point_set_infinity_jac(result);

    for (int i = 0; i < 256; i++) {
        if (get_bit(scalar, i)) {
            add_point_jac(result, result, &const_G_table[i]);
        }
    }
}

// FUNGSI KOMPRESI PUBKEY
__device__ void compress_pubkey(CompressedPubKey *compressed, const ECPoint *pubkey) {
    if (pubkey->infinity) {
        memset(compressed->data, 0, 33);
        return;
    }

    uint8_t prefix = (pubkey->y.data[0] & 1) ? 0x03 : 0x02;
    compressed->data[0] = prefix;

    for (int i = 0; i < BIGINT_WORDS; i++) {
        uint32_t word = pubkey->x.data[BIGINT_WORDS - 1 - i];
        compressed->data[1 + i*4] = (word >> 24) & 0xFF;
        compressed->data[2 + i*4] = (word >> 16) & 0xFF;
        compressed->data[3 + i*4] = (word >> 8) & 0xFF;
        compressed->data[4 + i*4] = word & 0xFF;
    }
}

// FUNGSI BLOOM FILTER DAN DP TABLE
__device__ uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

__device__ bool test_bit(const uint32_t* bloom_filter, uint64_t index) {
    uint64_t word_index = index / 32;
    uint32_t bit_index = index % 32;
    return (bloom_filter[word_index] >> bit_index) & 1;
}

__device__ bool check_bloom_filter(const uint32_t* bloom_filter, uint64_t bloom_size, uint64_t fp) {
    uint64_t h1 = splitmix64(fp);
    uint64_t h2 = splitmix64(h1);

    for (int j = 0; j < 4; j++) {
        uint64_t index = (h1 + j * h2) % bloom_size;
        if (!test_bit(bloom_filter, index)) {
            return false;
        }
    }
    return true;
}

__device__ uint64_t binary_search_dp_table(const DPEntry* dp_table, uint32_t dp_table_size, uint64_t fp) {
    int left = 0;
    int right = dp_table_size - 1;

    while (left <= right) {
        int mid = left + (right - left) / 2;
        uint64_t mid_fp = dp_table[mid].fp;

        if (mid_fp == fp) {
            return dp_table[mid].k_trap;
        } else if (mid_fp < fp) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    return 0;
}

// KERNEL UNTUK GENERATE TRAP TABLE (VERSI YANG BENAR)
extern "C"
__global__ void generate_trap_table_kernel(
    DPEntry* d_trap_table,
    uint32_t* d_bloom_filter,
    uint64_t trap_size,
    uint64_t bloom_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= trap_size) {
        return;
    }

    // k_trap = idx + 1 (karena mulai dari 1)
    uint64_t k_trap = (uint64_t)(idx + 1);

    // 1. Hitung k_trap * G menggunakan BigInt
    BigInt k_trap_bi;
    init_bigint(&k_trap_bi, 0);
    k_trap_bi.data[0] = (uint32_t)(k_trap & 0xFFFFFFFF);
    k_trap_bi.data[1] = (uint32_t)(k_trap >> 32);

    ECPointJac kG_jac;
    scalar_multiply_jac_precomputed(&kG_jac, &k_trap_bi);

    // 2. Konversi ke affine untuk fingerprint (LANGSUNG dari kG_jac)
    ECPoint kG_affine;
    jacobian_to_affine(&kG_affine, &kG_jac);

    // 3. Hitung fingerprint dari kG_affine.x
    uint64_t low64_x = kG_affine.x.data[0] | ((uint64_t)kG_affine.x.data[1] << 32);
    uint64_t y_parity = (kG_affine.y.data[0] & 1) ? 1 : 0;
    uint64_t fp = splitmix64(low64_x ^ y_parity);

    // 4. Simpan ke trap table
    d_trap_table[idx].fp = fp;
    d_trap_table[idx].k_trap = k_trap;

    // 5. Update bloom filter
    uint64_t h1 = splitmix64(fp);
    uint64_t h2 = splitmix64(h1);

    for (int j = 0; j < 4; j++) {
        uint64_t index = (h1 + j * h2) % bloom_size;
        uint64_t word_index = index / 32;
        uint32_t bit_index = index % 32;
        atomicOr(&d_bloom_filter[word_index], (1 << bit_index));
    }
}

// FUNGSI RANDOM NUMBER GENERATOR
__device__ uint64_t xorshift128plus(uint64_t* s) {
    uint64_t x = s[0];
    uint64_t y = s[1];
    s[0] = y;
    x ^= x << 23;
    s[1] = x ^ y ^ (x >> 17) ^ (y >> 26);
    return s[1] + y;
}

__device__ void init_random_state(uint64_t* state, uint64_t seed, uint64_t thread_id) {
    state[0] = seed + thread_id * 0x9E3779B97F4A7C15ULL;
    state[1] = seed ^ thread_id * 0xBF58476D1CE4E5B9ULL;
    if (state[0] == 0 && state[1] == 0) {
        state[0] = 0x9E3779B97F4A7C15ULL;
        state[1] = 0xBF58476D1CE4E5B9ULL;
    }
}

// FUNGSI UTILITAS MEMBANDINGKAN MEMORI
__device__ inline bool device_memcmp(const void *s1, const void *s2, size_t n) {
    const uint8_t *p1 = (const uint8_t *)s1;
    const uint8_t *p2 = (const uint8_t *)s2;
    for (size_t i = 0; i < n; ++i) {
        if (p1[i] != p2[i]) return false;
    }
    return true;
}

// FUNGSI UNTUK MEMBACA SCALAR BIGINT DARI MEMORI
__device__ void read_bigint_from_memory(BigInt *result, const uint32_t* scalar_memory, uint32_t scalar_index) {
    // Setiap scalar adalah 8 uint32 words
    for (int i = 0; i < BIGINT_WORDS; i++) {
        result->data[i] = scalar_memory[scalar_index * BIGINT_WORDS + i];
    }
}

// FUNGSI VERIFIKASI X-COORDINATE
__device__ bool verify_x_coordinate(const ECPoint *p1, const ECPoint *p2) {
    for (int i = 0; i < BIGINT_WORDS; i++) {
        if (p1->x.data[i] != p2->x.data[i]) {
            return false;
        }
    }
    return true;
}

// KERNEL UNTUK MITM REVERSE SEARCH (VERSI YANG BENAR)
extern "C"
__global__ void mitm_reverse_search_kernel(
    const uint32_t* d_scalar_list,  // Array of BigInts (8 uint32 per scalar)
    uint32_t num_scalars,
    const DPEntry* d_trap_table,
    const uint32_t* d_trap_bloom_filter,
    uint64_t trap_bloom_size,
    uint64_t trap_size,
    uint64_t seed,
    uint32_t num_experiments,
    uint32_t min_selected_scalars,
    uint32_t max_selected_scalars,
    BigInt* d_result,
    int* d_found_flag
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (*d_found_flag) return;
    if (idx >= num_experiments) return;

    // Initialize random state
    uint64_t random_state[2];
    init_random_state(random_state, seed, idx);

    // 1. GENERATE RANDOM N_STEP menggunakan BigInt
    uint32_t num_to_choose;
    uint32_t range = max_selected_scalars - min_selected_scalars + 1;
    if (range == 0) {
        return;
    }
    num_to_choose = (uint32_t)(xorshift128plus(random_state) % range) + min_selected_scalars;

    if (num_to_choose == 0) {
        return;
    }

    // Buat array untuk menyimpan indeks scalar yang dipilih
    uint32_t chosen_indices[128];
    BigInt n_step_bi;
    init_bigint(&n_step_bi, 0);

    for (uint32_t i = 0; i < num_to_choose; i++) {
        uint32_t scalar_idx;
        bool is_unique;

        do {
            scalar_idx = (uint32_t)(xorshift128plus(random_state) % num_scalars);
            is_unique = true;
            for (uint32_t j = 0; j < i; j++) {
                if (chosen_indices[j] == scalar_idx) {
                    is_unique = false;
                    break;
                }
            }
        } while (!is_unique);

        chosen_indices[i] = scalar_idx;

        BigInt current_scalar;
        read_bigint_from_memory(&current_scalar, d_scalar_list, scalar_idx);

        BigInt temp_sum;
        bigint_add_bigint(&temp_sum, &n_step_bi, &current_scalar);
        copy_bigint(&n_step_bi, &temp_sum);
    }

    if (is_zero(&n_step_bi)) return;

    // 2. HITUNG n_step * G
    ECPointJac nG_jac;
    scalar_multiply_jac_precomputed(&nG_jac, &n_step_bi);

    // 3. Negasi nG_jac to get -nG_jac
    ECPointJac neg_nG_jac;
    negate_point_jac(&neg_nG_jac, &nG_jac);

    // 4. Calculate T - n_step * G (ini adalah titik yang harus dicari di trap table)
    ECPointJac search_point_jac;
    add_point_jac(&search_point_jac, &const_target_jacobian, &neg_nG_jac);

    // 5. Konversi ke affine untuk fingerprint
    ECPoint search_point_affine;
    jacobian_to_affine(&search_point_affine, &search_point_jac);

    if (search_point_affine.infinity) return;

    // 6. Hitung fingerprint dari search_point_affine.x
    uint64_t low64_x = search_point_affine.x.data[0] | ((uint64_t)search_point_affine.x.data[1] << 32);
    uint64_t y_parity = (search_point_affine.y.data[0] & 1) ? 1 : 0;
    uint64_t fp = splitmix64(low64_x ^ y_parity);

    // 7. CEK DI TRAP TABLE (melalui bloom filter dulu)
    if (check_bloom_filter(d_trap_bloom_filter, trap_bloom_size, fp)) {
        uint64_t k_trap = binary_search_dp_table(d_trap_table, trap_size, fp);
        if (k_trap > 0) {
            // 8. VERIFIKASI TAMBAHAN: Hitung k_trap * G dan bandingkan x-coordinate
            BigInt k_trap_bi;
            init_bigint(&k_trap_bi, 0);
            k_trap_bi.data[0] = (uint32_t)(k_trap & 0xFFFFFFFF);
            k_trap_bi.data[1] = (uint32_t)(k_trap >> 32);

            ECPointJac kG_jac;
            scalar_multiply_jac_precomputed(&kG_jac, &k_trap_bi);

            ECPoint kG_affine;
            jacobian_to_affine(&kG_affine, &kG_jac);

            // Bandingkan x coordinate untuk verifikasi
            bool x_equal = true;
            for (int i = 0; i < BIGINT_WORDS; i++) {
                if (kG_affine.x.data[i] != search_point_affine.x.data[i]) {
                    x_equal = false;
                    break;
                }
            }

            if (x_equal) {
                // COLLISION VALID: T - n_step * G = k_trap * G
                if (atomicCAS(d_found_flag, 0, 1) == 0) {
                    // Simpan k_trap (posisi 0-31 bytes)
                    copy_bigint(&d_result[0], &k_trap_bi);

                    // Simpan n_step (posisi 32-63 bytes)
                    copy_bigint(&d_result[1], &n_step_bi);

                    // Simpan process info (posisi 64-95 bytes)
                    BigInt info;
                    init_bigint(&info, 0);
                    info.data[0] = (uint32_t)idx;
                    copy_bigint(&d_result[2], &info);
                }
            }
            return;
        }
    }
}

// KERNEL VERIFIKASI (diperbaiki)
extern "C"
__global__ void verify_collision_kernel(
    const BigInt* k_trap,
    const BigInt* n_step,
    const BigInt* target_pubkey_x,
    const BigInt* target_pubkey_y,
    int* verification_result
) {
    // 1. Hitung n_step * G
    ECPointJac nG_jac;
    scalar_multiply_jac_precomputed(&nG_jac, n_step);

    // 2. Hitung k_trap * G
    ECPointJac kG_jac;
    scalar_multiply_jac_precomputed(&kG_jac, k_trap);

    // 3. Hitung n_step * G + k_trap * G
    ECPointJac sum_jac;
    add_point_jac(&sum_jac, &nG_jac, &kG_jac);

    // 4. Konversi ke affine
    ECPoint sum_affine;
    jacobian_to_affine(&sum_affine, &sum_jac);

    // 5. Bandingkan dengan target
    bool x_equal = true;
    bool y_equal = true;

    for (int i = 0; i < BIGINT_WORDS; i++) {
        if (sum_affine.x.data[i] != target_pubkey_x->data[i]) {
            x_equal = false;
        }
        if (sum_affine.y.data[i] != target_pubkey_y->data[i]) {
            y_equal = false;
        }
    }

    *verification_result = (x_equal && y_equal) ? 1 : 0;
}
