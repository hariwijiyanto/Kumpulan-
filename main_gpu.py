import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
import time
import sys
import argparse
import struct
import random
import math
from datetime import datetime
from coincurve.keys import PrivateKey

def int_to_bigint_np(val):
    """Convert integer to BigInt numpy array (8 uint32 words)"""
    bigint_arr = np.zeros(8, dtype=np.uint32)
    for j in range(8):
        bigint_arr[j] = (val >> (32 * j)) & 0xFFFFFFFF
    return bigint_arr

def bigint_np_to_int(bigint_arr):
    """Convert BigInt numpy array to integer"""
    val = 0
    for j in range(8):
        val |= int(bigint_arr[j]) << (32 * j)
    return val

def decompress_pubkey(compressed):
    if len(compressed) != 33:
        raise ValueError("Compressed public key should be 33 bytes")
    prefix = compressed[0]
    x_bytes = compressed[1:]
    if prefix not in [0x02, 0x03]:
        raise ValueError("Invalid prefix for compressed public key")
    x_int = int.from_bytes(x_bytes, 'big')
    p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
    x3 = pow(x_int, 3, p)
    y_sq = (x3 + 7) % p
    y = pow(y_sq, (p + 1) // 4, p)
    is_even = (y % 2 == 0)
    if (prefix == 0x02 and not is_even) or (prefix == 0x03 and is_even):
        y = p - y
    return x_int, y

def init_secp256k1_constants(mod):
    # Initialize prime p
    p_data = np.array([
        0xFFFFFC2F, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF,
        0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
    ], dtype=np.uint32)
    const_p_gpu = mod.get_global("const_p")[0]
    cuda.memcpy_htod(const_p_gpu, p_data)

    # Initialize order n
    n_data = np.array([
        0xD0364141, 0xBFD25E8C, 0xAF48A03B, 0xBAAEDCE6,
        0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
    ], dtype=np.uint32)
    const_n_gpu = mod.get_global("const_n")[0]
    cuda.memcpy_htod(const_n_gpu, n_data)

    # Initialize generator point G
    g_x = np.array([
        0x16F81798, 0x59F2815B, 0x2DCE28D9, 0x029BFCDB,
        0xCE870B07, 0x55A06295, 0xF9DCBBAC, 0x79BE667E
    ], dtype=np.uint32)
    g_y = np.array([
        0xFB10D4B8, 0x9C47D08F, 0xA6855419, 0xFD17B448,
        0x0E1108A8, 0x5DA4FBFC, 0x26A3C465, 0x483ADA77
    ], dtype=np.uint32)
    g_z = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint32)
    g_infinity = np.array([False], dtype=np.bool_)

    ecpoint_jac_dtype = np.dtype([
        ('X', np.uint32, 8),
        ('Y', np.uint32, 8),
        ('Z', np.uint32, 8),
        ('infinity', np.bool_)
    ])
    g_jac = np.zeros(1, dtype=ecpoint_jac_dtype)
    g_jac['X'], g_jac['Y'], g_jac['Z'], g_jac['infinity'] = g_x, g_y, g_z, g_infinity
    const_G_gpu = mod.get_global("const_G_jacobian")[0]
    cuda.memcpy_htod(const_G_gpu, g_jac)

def run_precomputation(mod):
    precompute_kernel = mod.get_function("precompute_G_table_kernel")
    precompute_kernel(block=(1, 1, 1))
    cuda.Context.synchronize()

def get_scalar_list(total_scalars: int, start_exponent: int) -> list:
    """Generate scalar list: 2**start_exponent to 2**(start_exponent + total_scalars - 1) as BigInts"""
    full_scalars = []
    for i in range(start_exponent, start_exponent + total_scalars):
        scalar_val = 2**i
        bigint_scalar = int_to_bigint_np(scalar_val)
        full_scalars.append(bigint_scalar)
    return full_scalars

def calculate_optimal_parameters():
    """Calculate optimal parameters based on memory and performance"""
    # Default optimal values
    params = {
        'trap_size': 2**25,  # 16.7 million - larger trap table
        'total_scalars': 25,  # More scalars for more combinations
        'start_exponent': 1,
        'experiments_per_batch': 2**18,  # 16.7 million per batch
        'num_batches': 2**15  # Run batches
    }

    return params

def verify_collision_with_coincurve(private_key_int: int, target_pubkey_hex: str) -> bool:
    """Verifies the collision using coincurve to detect false positives."""
    try:
        priv_key_bytes = private_key_int.to_bytes(32, 'big')
        private_key = PrivateKey(priv_key_bytes)
        derived_pubkey_bytes = private_key.public_key.format(compressed=True)
        target_pubkey_bytes = bytes.fromhex(target_pubkey_hex)

        return derived_pubkey_bytes == target_pubkey_bytes
    except Exception as e:
        print(f"[!] Verification failed with coincurve: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description='MITM Attack CUDA - Versi Terbalik (Optimized)',
    )

    parser.add_argument('--pubkey', type=str, required=True,
                       help='Compressed public key (66 chars hex)')
    parser.add_argument('--trap-size', type=int, default=0,
                       help='Trap table size (0 for auto-calculate)')
    parser.add_argument('--total-scalars', type=int, default=0,
                       help='Total number of scalars (0 for auto-calculate)')
    parser.add_argument('--start-exponent', type=int, default=1,
                       help='Starting exponent for scalars (default: 1)')
    parser.add_argument('--batches', type=int, default=2**100,
                       help='Number of batches to run')
    parser.add_argument('--batch-size', type=int, default=0,
                       help='Experiments per batch (0 for auto-calculate)')
    parser.add_argument('--min-selected-scalars', type=int, default=1,
                       help='Minimum number of scalars to select for n_step (default: 1)')
    parser.add_argument('--max-selected-scalars', type=int, default=0,
                       help='Maximum number of scalars to select for n_step (0 for total_scalars)')

    args = parser.parse_args()

    if len(args.pubkey) != 66:
        print("[ERROR] Compressed pubkey must be 66 characters")
        return

    print("=" * 70)
    print("MITM ATTACK CUDA - VERSI TERBALIK (OPTIMIZED)")
    print("=" * 70)

    # Calculate optimal parameters
    optimal = calculate_optimal_parameters()

    if args.trap_size == 0:
        args.trap_size = optimal['trap_size']
    if args.total_scalars == 0:
        args.total_scalars = optimal['total_scalars']
    if args.batch_size == 0:
        args.batch_size = optimal['experiments_per_batch']

    # Adjust min/max selected scalars based on total_scalars
    if args.max_selected_scalars == 0:
        args.max_selected_scalars = args.total_scalars
    # Ensure min <= max
    if args.min_selected_scalars > args.max_selected_scalars:
        print(f"[WARNING] min_selected_scalars ({args.min_selected_scalars}) cannot be greater than max_selected_scalars ({args.max_selected_scalars}). Adjusting max_selected_scalars to {args.min_selected_scalars}.")
        args.max_selected_scalars = args.min_selected_scalars
    # Ensure max does not exceed total_scalars
    if args.max_selected_scalars > args.total_scalars:
        print(f"[WARNING] max_selected_scalars ({args.max_selected_scalars}) cannot exceed total_scalars ({args.total_scalars}). Adjusting max_selected_scalars to {args.total_scalars}.")
        args.max_selected_scalars = args.total_scalars


    print(f"[*] Configuration:")
    print(f"    Trap table size: {args.trap_size:,} ({math.log2(args.trap_size):.1f} bits)")
    print(f"    Total scalars: {args.total_scalars}")
    print(f"    Start exponent: {args.start_exponent}")
    print(f"    Min selected scalars: {args.min_selected_scalars}")
    print(f"    Max selected scalars: {args.max_selected_scalars}")
    print(f"    Batches: {args.batches}")
    print(f"    Batch size: {args.batch_size:,} experiments")

    # Load CUDA code
    try:
        with open('main_mitm.cu', 'r') as f:
            full_cuda_code = f.read()
    except FileNotFoundError:
        print("[!] FATAL: File 'main_mitm.cu' tidak ditemukan.")
        sys.exit(1)

    # Compile CUDA code
    print("[*] Compiling CUDA code...")
    mod = SourceModule(full_cuda_code, no_extern_c=False, options=['-std=c++11', '-arch=sm_75', '-O3'])

    # Decompress target pubkey
    print(f"[*] Target: {args.pubkey[:16]}...{args.pubkey[-16:]}")
    try:
        pubkey_bytes = bytes.fromhex(args.pubkey)
        target_x, target_y = decompress_pubkey(pubkey_bytes)
        print(f"[*] Target decompressed")

        # Set target point in GPU constant memory
        ecpoint_jac_dtype = np.dtype([
            ('X', np.uint32, 8),
            ('Y', np.uint32, 8),
            ('Z', np.uint32, 8),
            ('infinity', np.bool_)
        ])
        target_jac = np.zeros(1, dtype=ecpoint_jac_dtype)
        target_jac['X'] = int_to_bigint_np(target_x)
        target_jac['Y'] = int_to_bigint_np(target_y)
        target_jac['Z'] = int_to_bigint_np(1)
        target_jac['infinity'] = False

        const_target_gpu = mod.get_global("const_target_jacobian")[0]
        cuda.memcpy_htod(const_target_gpu, target_jac)

    except Exception as e:
        print(f"[ERROR] Failed to process target: {e}")
        return

    # Initialize constants
    init_secp256k1_constants(mod)

    # Precomputation
    print("[*] Running precomputation...")
    run_precomputation(mod)

    # Generate scalar list as BigInts
    scalar_bigints = get_scalar_list(args.total_scalars, args.start_exponent)

    print(f"\n[*] Scalar list for n_step:")
    print(f"    Range: 2^{args.start_exponent} to 2^{args.start_exponent + args.total_scalars - 1}")
    print(f"    Number of scalars: {len(scalar_bigints)}")
    print(f"    Total possible subsets: 2^{len(scalar_bigints):,} (with no min/max restrictions)")

    # Convert scalar list to GPU (as array of BigInts)
    scalar_array = np.array(scalar_bigints, dtype=np.uint32).reshape(-1, 8)
    d_scalar_list = cuda.mem_alloc(scalar_array.size * 4)  # 4 bytes per uint32
    cuda.memcpy_htod(d_scalar_list, scalar_array)

    # Generate trap table
    print("\n" + "=" * 70)
    print("GENERATING TRAP TABLE")
    print("=" * 70)

    start_time = time.time()

    # Kernel untuk generate trap table
    generate_trap_kernel = mod.get_function("generate_trap_table_kernel")

    # Allocate memory untuk trap table
    trap_entry_dtype = np.dtype([('fp', np.uint64), ('k_trap', np.uint64)])
    d_trap_table = cuda.mem_alloc(args.trap_size * trap_entry_dtype.itemsize)

    # Calculate optimal bloom filter size
    false_positive_rate = 0.0001
    bloom_size = int(- (args.trap_size * math.log(false_positive_rate)) / (math.log(2) ** 2))
    bloom_size = int(2 ** math.ceil(math.log2(bloom_size)))  # Round up to power of 2

    d_bloom_filter = cuda.mem_alloc((bloom_size + 31) // 32 * 4)
    cuda.memset_d32(d_bloom_filter, 0, (bloom_size + 31) // 32)

    # Run trap table generation
    block_size = 256
    grid_size = (args.trap_size + block_size - 1) // block_size

    print(f"[*] Generating {args.trap_size:,} trap entries on GPU...")
    generate_trap_kernel(
        d_trap_table, d_bloom_filter,
        np.uint64(args.trap_size), np.uint64(bloom_size),
        block=(block_size, 1, 1), grid=(grid_size, 1)
    )
    cuda.Context.synchronize()

    trap_time = time.time() - start_time
    print(f"[*] Generated {args.trap_size:,} entries in {trap_time:.3f}s ({args.trap_size/trap_time:,.0f}/sec)")
    print(f"[*] Bloom filter size: {bloom_size:,} bits ({bloom_size/8/1024/1024:.2f} MB)")

    # Sort trap table
    print("[*] Sorting trap table...")
    trap_table_host = np.zeros(args.trap_size, dtype=trap_entry_dtype)
    cuda.memcpy_dtoh(trap_table_host, d_trap_table)
    trap_table_host.sort(order='fp')
    cuda.memcpy_htod(d_trap_table, trap_table_host)

    # Setup search kernel
    mitm_reverse_kernel = mod.get_function("mitm_reverse_search_kernel")

    # Allocate memory untuk hasil
    d_result = cuda.mem_alloc(96)  # k_trap, n_step, info
    d_found_flag = cuda.mem_alloc(4)

    # Setup kernel parameters
    block_size = 256
    grid_size = (args.batch_size + block_size - 1) // block_size

    print("\n" + "=" * 70)
    print("STARTING MITM REVERSE SEARCH")
    print("=" * 70)

    print(f"[*] Search configuration:")
    print(f"    Threads per block: {block_size}")
    print(f"    Grid size: {grid_size}")
    print(f"    Total threads per batch: {block_size * grid_size:,}")
    print(f"    Experiments per batch: {args.batch_size:,}")
    print(f"    Total batches: {args.batches}")
    print(f"    Total experiments: {args.batches * args.batch_size:,}")

    total_experiments = 0
    total_time = 0
    found = False
    k_trap = 0
    n_step_bigint = None
    private_key_verified = False

    start_search_time = time.time()

    try:
        for batch in range(args.batches):
            if found and private_key_verified:
                break

            batch_start = time.time()

            # Reset found flag
            cuda.memset_d32(d_found_flag, 0, 1)

            # Set unique seed for each batch
            seed = np.uint64(time.time() * 1000 + batch)

            # Run kernel
            mitm_reverse_kernel(
                d_scalar_list, np.uint32(len(scalar_bigints)),
                d_trap_table, d_bloom_filter,
                np.uint64(bloom_size), np.uint64(args.trap_size),
                seed,
                np.uint32(args.batch_size),
                np.uint32(args.min_selected_scalars), # New parameter
                np.uint32(args.max_selected_scalars), # New parameter
                d_result, d_found_flag,
                block=(block_size, 1, 1), grid=(grid_size, 1)
            )
            cuda.Context.synchronize()

            # Check if found
            found_flag_host = np.zeros(1, dtype=np.int32)
            cuda.memcpy_dtoh(found_flag_host, d_found_flag)

            total_experiments += args.batch_size
            batch_time = time.time() - batch_start
            total_time = time.time() - start_search_time

            if found_flag_host[0]:
                # Read result
                result_buffer = np.zeros(24, dtype=np.uint32)
                cuda.memcpy_dtoh(result_buffer, d_result)

                k_trap_np = result_buffer[:8]
                n_step_np = result_buffer[8:16]

                k_trap = bigint_np_to_int(k_trap_np)
                n_step_bigint = n_step_np
                found = True

                # Calculate private key for verification
                N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
                n_step_val = bigint_np_to_int(n_step_bigint)
                private_key = (n_step_val - k_trap) % N

                print(f"\n[+] CANDIDATE PRIVATE KEY FOUND: {hex(private_key)}")

                # Verify with coincurve
                print(f"[+] COLLISION FOUND in batch {batch + 1}! Verifying with coincurve...")
                private_key_verified = verify_collision_with_coincurve(private_key, args.pubkey)

                if private_key_verified:
                    print("[+] Coincurve verification SUCCESS!")
                    print(f"[+] VERIFIED PRIVATE KEY: {hex(private_key)}")
                    break
                else:
                    print(f"[!] Coincurve verification FAILED for {hex(private_key)}. This might be a false positive. Continuing search...")
                    # Reset found flag to continue searching if it was a false positive
                    found = False
                    cuda.memset_d32(d_found_flag, 0, 0) # Clear the flag on GPU to allow new collisions

            # Calculate statistics
            speed = args.batch_size / batch_time if batch_time > 0 else 0
            avg_speed = total_experiments / total_time if total_time > 0 else 0
            estimated_total = args.batches * args.batch_size
            progress = (total_experiments / estimated_total) * 100

            # Calculate probability
            # Probability of collision = 1 - (1 - trap_size/N)^experiments
            N_fp_space = 2**64  # 64-bit fingerprint space
            prob_no_collision = (1 - args.trap_size/N_fp_space) ** total_experiments
            prob_collision = (1 - prob_no_collision) * 100

            progress_str = (
                f"\r[Batch {batch+1:3d}/{args.batches}] "
                f"Progress: {progress:6.2f}% | "
                f"Experiments: {total_experiments:,} | "
                f"Batch speed: {speed:,.0f} exp/s | "
                f"Avg speed: {avg_speed:,.0f} exp/s | "
                f"Collision probability: {prob_collision:.2f}%"
            )
            sys.stdout.write(progress_str.ljust(140))
            sys.stdout.flush()

    except KeyboardInterrupt:
        print(f"\n\n[!] Search interrupted by user")
    except Exception as e:
        print(f"\n\n[!] Error: {e}")
        import traceback
        traceback.print_exc()

    print()

    if found and private_key_verified:
        # Calculate private key: K = n_step - k_trap mod N
        N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
        n_step_val = bigint_np_to_int(n_step_bigint)
        private_key = (n_step_val - k_trap) % N

        print("\n" + "=" * 70)
        print("[+] SUCCESS! COLLISION FOUND!")
        print("=" * 70)
        print(f"    Private Key: {hex(private_key)}")
        print(f"    n_step: {n_step_val:,}")
        print(f"    k_trap: {k_trap:,}")
        print(f"    Calculation: {n_step_val} - {k_trap} = {private_key} mod N")
        print(f"    Total experiments: {total_experiments:,}")
        print(f"    Total search time: {total_time:.2f}s")
        print(f"    Average speed: {total_experiments/total_time:,.0f} exp/s")

        # Simple verification
        print(f"\n[*] Performing verification...")
        if (n_step_val > 0 and k_trap > 0):
            print(f"    n_step valid: \u2713")
            print(f"    k_trap valid: \u2713")
            print(f"    Private key within range: \u2713")

            # Save results
            timestamp = int(time.time())
            filename = f"found_key_mitm_{timestamp}.txt"
            with open(filename, "w") as f:
                f.write(f"MITM Attack CUDA - Reverse Version\n")
                f.write(f"==================================\n")
                f.write(f"Public Key: {args.pubkey}\n")
                f.write(f"Private Key (hex): {hex(private_key)}\n")
                f.write(f"Private Key (dec): {private_key}\n")
                f.write(f"n_step: {n_step_val}\n")
                f.write(f"k_trap: {k_trap}\n")
                f.write(f"Formula: n_step * G = T + k_trap * G\n")
                f.write(f"Calculation: K = n_step - k_trap mod N\n")
                f.write(f"Total experiments: {total_experiments}\n")
                f.write(f"Total search time: {total_time:.2f}s\n")
                f.write(f"Average speed: {total_experiments/total_time:.0f} exp/s\n")
                f.write(f"Trap table size: {args.trap_size}\n")
                f.write(f"Scalar list size: {len(scalar_bigints)}\n")
                f.write(f"Min selected scalars: {args.min_selected_scalars}\n")
                f.write(f"Max selected scalars: {args.max_selected_scalars}\n")
                f.write(f"Found at: {datetime.now().isoformat()}\n")

            print(f"\n[+] Results saved to {filename}")
        else:
            print(f"    [!] Verification warning: values may be invalid")
    else:
        print(f"\n[-] Private key not found after {total_experiments:,} experiments")
        print(f"    Total search time: {total_time:.2f}s")
        print(f"    Average speed: {total_experiments/total_time:,.0f} exp/s")

        # Calculate remaining probability
        if total_experiments > 0:
            N_fp_space = 2**64
            prob_no_collision = (1 - args.trap_size/N_fp_space) ** total_experiments
            prob_collision = (1 - prob_no_collision) * 100
            print(f"    Collision probability achieved: {prob_collision:.2f}%")

            # Estimate remaining experiments needed for 50% probability
            target_prob = 0.5
            remaining = int(math.log(1 - target_prob) / math.log(1 - args.trap_size/N_fp_space) - total_experiments)
            if remaining > 0:
                print(f"    Estimated experiments needed for 50% chance: {remaining:,}")

    print("\n" + "=" * 70)
    print("PROGRAM COMPLETED")
    print("=" * 70)

if __name__ == '__main__':
    import traceback
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[!] Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        traceback.print_exc()
        sys.exit(1)
