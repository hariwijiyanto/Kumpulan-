#!/usr/bin/env python3
"""
PENCARIAN PRIVATE KEY UNTUK H160 (HASH160)
Versi: Dengan pemilihan N skalar acak dari seluruh pool.
"""

import time
import sys
import argparse
import random
import multiprocessing as mp
import os
from typing import List, Tuple, Optional
from datetime import datetime

import coincurve
from Crypto.Hash import RIPEMD160
import hashlib

# =================================================================
# KONSTANTA
# =================================================================
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
G_COMPRESSED = bytes.fromhex("0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798")

# =================================================================
# KONFIGURASI DEFAULT
# =================================================================
DEFAULT_START_EXPONENT = 0  # Default to 2^0
DEFAULT_MAX_EXPONENT = 71   # Default to 2^71
DEFAULT_MIN_SCALARS_TO_PICK = 32 # Default minimum scalars to pick
DEFAULT_MAX_SCALARS_TO_PICK = 40 # Default maximum scalars to pick

# =================================================================
# FUNGSI UTILITAS
# =================================================================
def hash160(public_key_bytes: bytes) -> bytes:
    """Hitung HASH160 (SHA256 + RIPEMD160) dari public key"""
    sha256_hash = hashlib.sha256(public_key_bytes).digest()
    ripemd160 = RIPEMD160.new()
    ripemd160.update(sha256_hash)
    return ripemd160.digest()

def pubkey_to_h160(public_key_hex: str) -> bytes:
    """Konversi public key hex ke HASH160"""
    pubkey_bytes = bytes.fromhex(public_key_hex)
    return hash160(pubkey_bytes)

def get_all_possible_scalars(
    start_exponent: int,
    max_exponent: int
) -> List[int]:
    """Generate semua skalar yang mungkin dari 2^start_exponent sampai 2^max_exponent"""
    if max_exponent < start_exponent:
        raise ValueError("max_exponent cannot be less than start_exponent")
    return [(2**i) for i in range(start_exponent, max_exponent + 1)]

# =================================================================
# WORKER FUNCTION
# =================================================================
def worker_search_h160(
    target_h160: bytes,
    all_possible_scalars: List[int],
    result_queue: mp.Queue,
    total_completed,
    process_id: int,
    max_experiments: int,
    min_scalars_to_pick: int,
    max_scalars_to_pick: int
):
    """
    Worker: Mencari n_step dimana H160(n_step * G) = target_h160
    Akan memilih `min_scalars_to_pick`-`max_scalars_to_pick` skalar unik dari all_possible_scalars untuk setiap eksperimen.
    """
    random.seed(os.getpid() + process_id + int(time.time() * 1234567890))
    num_available_scalars = len(all_possible_scalars)

    if num_available_scalars == 0:
        result_queue.put({'found': False, 'process_id': process_id})
        return

    for n_exp in range(max_experiments):
        # 1. Tentukan secara acak berapa banyak skalar yang akan dipilih
        # Pastikan tidak melebihi jumlah skalar yang tersedia
        num_scalars_to_pick = random.randint(min_scalars_to_pick, max_scalars_to_pick)
        if num_scalars_to_pick > num_available_scalars:
            num_scalars_to_pick = num_available_scalars

        # 2. Pilih skalar-skalar unik secara acak dari seluruh daftar
        chosen_scalars = random.sample(all_possible_scalars, num_scalars_to_pick)

        n_step = 0
        for scalar_val in chosen_scalars:
            n_step = (n_step + scalar_val) % N

        if n_step == 0:
            continue

        # Hitung n_step * G
        n_step_bytes = n_step.to_bytes(32, 'big')
        try:
            nG_pubkey = coincurve.PublicKey.from_valid_secret(n_step_bytes)
        except ValueError: # Tangani kasus n_step adalah 0 atau tidak valid
            continue

        nG_compressed = nG_pubkey.format(compressed=True)
        current_h160 = hash160(nG_compressed)

        # Bandingkan dengan target H160
        if current_h160 == target_h160:
            result_queue.put(
                {
                    'found': True,
                    'n_step': n_step,
                    'experiments': n_exp + 1,
                    'process_id': process_id,
                    'public_key': nG_compressed.hex()
                }
            )
            return

        # Update progress
        if n_exp % 1000 == 0:
            with total_completed.get_lock():
                total_completed.value += 1000

        # Tambahkan print output skalar setiap 10000 iterasi
        if n_exp % 500000 == 0 and n_exp > 0:
            chosen_exponents = sorted([s.bit_length() - 1 for s in chosen_scalars], reverse=True)
            sys.stdout.write(f"\r[Process {process_id}] Exp {n_exp:,}: Chosen Scalars (Exponents): {chosen_exponents}\n")
            sys.stdout.flush()


    result_queue.put({'found': False, 'process_id': process_id})

# =================================================================
# VERIFICATION H160
# =================================================================
def verify_h160(public_key_hex: str, target_h160: bytes) -> bool:
    """Verifikasi apakah public key menghasilkan H160 yang sesuai"""
    pubkey_bytes = bytes.fromhex(public_key_hex)
    calculated_h160 = hash160(pubkey_bytes)
    return calculated_h160 == target_h160

# =================================================================
# MAIN SEARCH FUNCTION
# =================================================================
def search_private_key_h160(
    target_h160: bytes,
    all_possible_scalars: List[int],
    num_processes: int,
    max_experiments_per_process: int,
    min_scalars_to_pick_arg: int,
    max_scalars_to_pick_arg: int
) -> Optional[Tuple[int, str, int, int]]:
    """Fungsi pencarian utama untuk H160"""
    print(f"[*] Starting search for H160 with {num_processes} processes")
    print(f"[*] Target H160: {target_h160.hex()}")
    print(f"[*] Max experiments per process: {max_experiments_per_process:,}")
    print(f"[*] Scalars to pick per experiment: {min_scalars_to_pick_arg}-{max_scalars_to_pick_arg}")

    result_queue = mp.Queue()
    total_completed = mp.Value('i', 0)

    processes = []
    for i in range(num_processes):
        p = mp.Process(
            target=worker_search_h160,
            args=(
                target_h160,
                all_possible_scalars,
                result_queue,
                total_completed,
                i,
                max_experiments_per_process,
                min_scalars_to_pick_arg,
                max_scalars_to_pick_arg
            )
        )
        processes.append(p)
        p.start()

    search_start = time.time()
    found = False
    result = None

    try:
        while True:
            if not result_queue.empty():
                res = result_queue.get()
                if res['found']:
                    found = True
                    result = res
                    break

            alive_count = sum(1 for p in processes if p.is_alive())
            if alive_count == 0 and result_queue.empty():
                break

            elapsed = time.time() - search_start
            current = total_completed.value
            speed = current / elapsed if elapsed > 0 else 0

            # Pastikan progress bar dan output scalars tidak saling menumpuk
            # dengan mencetak newline setelah output scalars
            sys.stdout.write(
                f"\r[Progress] Experiments: {current:,} | "
                f"Speed: {speed:,.0f} it/s | "
                f"Active: {alive_count}/{num_processes}"
            )
            sys.stdout.flush()

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n[!] Search interrupted")

    for p in processes:
        p.terminate()
        p.join()

    if found and result:
        print(f"\n[*] Found in process {result['process_id']} "
              f"after {result['experiments']:,} experiments")
        return (result['n_step'], result['public_key'],
                result['experiments'], result['process_id'])

    return None

# =================================================================
# MAIN PROGRAM
# =================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Pencarian Private Key untuk HASH160',
    )

    parser.add_argument('--h160', type=str, required=True,
                       help='Target HASH160 (40 chars hex)')
    parser.add_argument('--max_experiments', type=int, default=2**100,
                       help='Max experiments per process')
    parser.add_argument('--processes', type=int, default=mp.cpu_count(),
                       help='Number of CPU processes')

    parser.add_argument('--start_exponent', type=int, default=DEFAULT_START_EXPONENT,
                       help=f'Starting exponent for scalars (default: {DEFAULT_START_EXPONENT} for 2^{DEFAULT_START_EXPONENT})')
    parser.add_argument('--max_exponent', type=int, default=DEFAULT_MAX_EXPONENT,
                       help=f'Maximum exponent for scalars (default: {DEFAULT_MAX_EXPONENT} for 2^{DEFAULT_MAX_EXPONENT})')
    parser.add_argument('--min_scalars_to_pick', type=int, default=DEFAULT_MIN_SCALARS_TO_PICK,
                       help=f'Minimum number of scalars to pick (default: {DEFAULT_MIN_SCALARS_TO_PICK})')
    parser.add_argument('--max_scalars_to_pick', type=int, default=DEFAULT_MAX_SCALARS_TO_PICK,
                       help=f'Maximum number of scalars to pick (default: {DEFAULT_MAX_SCALARS_TO_PICK})')

    args = parser.parse_args()

    if len(args.h160) != 40:
        print("[ERROR] HASH160 harus 40 karakter hex")
        return

    try:
        target_h160 = bytes.fromhex(args.h160)
    except ValueError:
        print("[ERROR] Format HASH160 tidak valid")
        return

    print("=" * 70)
    print("PENCARIAN PRIVATE KEY UNTUK HASH160")
    print("=" * 70)

    print(f"[*] Target H160: {args.h160}")

    # Generate semua skalar yang mungkin
    all_possible_scalars = get_all_possible_scalars(
        args.start_exponent, args.max_exponent
    )

    print(f"\n[*] Konfigurasi skalar:")
    print(f"    Start Exponent: {args.start_exponent}")
    print(f"    Max Exponent: {args.max_exponent}")
    print(f"    Total Skalar Tersedia: {len(all_possible_scalars)}")
    print(f"[*] Setiap eksperimen akan memilih {args.min_scalars_to_pick}-{args.max_scalars_to_pick} skalar unik dari total skalar yang tersedia.")

    print("\n" + "=" * 70)
    print("MEMULAI PENCARIAN")
    print("=" * 70)

    search_start = time.time()

    result = search_private_key_h160(
        target_h160=target_h160,
        all_possible_scalars=all_possible_scalars,
        num_processes=args.processes,
        max_experiments_per_process=args.max_experiments,
        min_scalars_to_pick_arg=args.min_scalars_to_pick,
        max_scalars_to_pick_arg=args.max_scalars_to_pick
    )

    total_time = time.time() - search_start

    if result:
        n_step, public_key, experiments, process_id = result
        private_key = n_step % N

        print("\n" + "=" * 70)
        print("[+] PRIVATE KEY DITEMUKAN!")
        print("=" * 70)
        print(f"    Private Key (hex): {hex(private_key)}")
        print(f"    Private Key (dec): {private_key:,}")
        print(f"    n_step: {n_step:,}")
        print(f"    Public Key: {public_key}")
        print(f"    Process: {process_id}")
        print(f"    Experiments: {experiments:,}")
        print(f"    Search time: {total_time:.2f}s")

        if verify_h160(public_key, target_h160):
            print("[+] VERIFIKASI BERHASIL! H160 cocok.")
            calculated_h160 = hash160(bytes.fromhex(public_key))
            print(f"[+] H160 ditemukan: {calculated_h160.hex()}")

            filename = f"found_key_h160_{int(time.time())}.txt"
            with open(filename, "w") as f:
                f.write(f"Target H160: {args.h160}\n")
                f.write(f"Private Key (hex): {hex(private_key)}\n")
                f.write(f"Private Key (dec): {private_key}\n")
                f.write(f"Public Key: {public_key}\n")
                f.write(f"n_step: {n_step}\n")
                f.write(f"Process: {process_id}\n")
                f.write(f"Experiments: {experiments}\n")
                f.write(f"Search time: {total_time:.2f}s\n")
                f.write(f"Found at: {datetime.now().isoformat()}\n")
                f.write(f"Verification: H160 matched\n")

            print(f"[*] Hasil disimpan ke {filename}")
        else:
            print("[-] VERIFIKASI GAGAL! H160 tidak cocok.")
    else:
        print(f"\n[-] Private key tidak ditemukan setelah {total_time:.2f}s")

    print("\n" + "=" * 70)
    print("PROGRAM SELESAI")
    print("=" * 70)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n[!] Diinterupsi oleh pengguna")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
