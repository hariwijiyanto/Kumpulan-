#!/usr/bin/env python3
"""
SIMPLE H160/RIPEMD160 SEARCH TOOL
"""

import time
import sys
import argparse
import random
import multiprocessing as mp
import os
import hashlib
from typing import List, Optional

import coincurve
from Crypto.Hash import RIPEMD160

# =================================================================
# KONSTANTA
# =================================================================
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141

# =================================================================
# LOAD SCALAR FILES
# =================================================================
def load_scalars(filepaths: List[str]) -> List[List[int]]:
    """Load scalars from files"""
    scalar_groups = []

    for fp in filepaths:
        if not os.path.exists(fp):
            print(f"[!] File not found: {fp}")
            continue

        group = []
        with open(fp, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    if line.startswith('0x'):
                        val = int(line, 16)
                    else:
                        val = int(line)
                    group.append(val % N)
                except:
                    continue

        if group:
            scalar_groups.append(group)
            print(f"[+] Loaded {len(group)} scalars from {os.path.basename(fp)}")

    if not scalar_groups:
        raise ValueError("No scalars loaded!")

    total = 1
    for g in scalar_groups:
        total *= len(g)

    print(f"[+] Total combinations: {total:,}")
    return scalar_groups

# =================================================================
# HASH FUNCTIONS
# =================================================================
def pubkey_to_h160(pubkey_bytes: bytes) -> str:
    """Convert public key to HASH160"""
    # SHA256
    sha = hashlib.sha256(pubkey_bytes).digest()
    # RIPEMD160
    ripemd = RIPEMD160.new()
    ripemd.update(sha)
    return ripemd.hexdigest()

def generate_private_key(scalar_groups: List[List[int]]) -> int:
    """Generate private key from scalar groups"""
    priv = 0
    for group in scalar_groups:
        priv = (priv + random.choice(group)) % N
    return priv

# =================================================================
# WORKER FUNCTION
# =================================================================
def worker_search(
    target_h160: str,
    scalar_groups: List[List[int]],
    result_queue: mp.Queue,
    counter,
    worker_id: int,
    max_tries: int
):
    """Worker process for searching"""
    random.seed(worker_id + int(time.time()*1234567890))

    for attempt in range(max_tries):
        # Generate private key
        priv = (generate_private_key(scalar_groups)) - 1
        if priv == 0:
            continue

        # Generate public key
        try:
            priv_bytes = priv.to_bytes(32, 'big')
            pubkey = coincurve.PublicKey.from_valid_secret(priv_bytes)
            pubkey_bytes = pubkey.format()  # Compressed format
        except:
            continue

        # Calculate H160
        h160 = pubkey_to_h160(pubkey_bytes)

        # Check match
        if h160 == target_h160:
            result_queue.put({
                'found': True,
                'private': priv,
                'public': pubkey_bytes.hex(),
                'h160': h160,
                'attempts': attempt + 1,
                'worker': worker_id
            })
            return

        # Update counter every 100 attempts
        if attempt % 100 == 0:
            with counter.get_lock():
                counter.value += 100

    result_queue.put({'found': False, 'worker': worker_id})

# =================================================================
# MAIN SEARCH FUNCTION
# =================================================================
def search_h160(
    target_h160: str,
    scalar_groups: List[List[int]],
    num_workers: int,
    max_per_worker: int
) -> Optional[dict]:
    """Main search function"""
    print(f"\n[+] Target H160: {target_h160}")
    print(f"[+] Workers: {num_workers}")
    print(f"[+] Attempts per worker: {max_per_worker:,}")
    print(f"[+] Total attempts: {num_workers * max_per_worker:,}")
    print("-" * 50)

    result_queue = mp.Queue()
    counter = mp.Value('i', 0)

    # Start workers
    workers = []
    for i in range(num_workers):
        w = mp.Process(
            target=worker_search,
            args=(target_h160, scalar_groups, result_queue, counter, i, max_per_worker)
        )
        workers.append(w)
        w.start()

    start_time = time.time()
    found = False
    result = None

    try:
        while not found:
            # Check results
            if not result_queue.empty():
                res = result_queue.get()
                if res['found']:
                    found = True
                    result = res
                    break

            # Check if all workers are done
            alive = sum(1 for w in workers if w.is_alive())
            if alive == 0 and result_queue.empty():
                break

            # Display progress
            elapsed = time.time() - start_time
            attempts = counter.value
            speed = attempts / elapsed if elapsed > 0 else 0

            sys.stdout.write(
                f"\r[Progress] Attempts: {attempts:,} | "
                f"Speed: {speed:,.0f}/s | "
                f"Workers: {alive}/{num_workers} | "
                f"Time: {elapsed:.1f}s"
            )
            sys.stdout.flush()

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n[!] Stopped by user")

    finally:
        # Cleanup
        for w in workers:
            if w.is_alive():
                w.terminate()
            w.join()

    print()  # New line after progress
    return result

# =================================================================
# MAIN
# =================================================================
def main():
    parser = argparse.ArgumentParser(description='Simple H160 Search Tool')
    parser.add_argument('--h160', type=str, required=True, help='Target HASH160 (hex)')
    parser.add_argument('--files', nargs='+', required=True, help='Scalar files')
    parser.add_argument('--workers', type=int, default=mp.cpu_count(), help='Number of workers')
    parser.add_argument('--attempts', type=int, default=10000000000, help='Attempts per worker')

    args = parser.parse_args()

    print("=" * 60)
    print("H160/RIPEMD160 SEARCH TOOL")
    print("=" * 60)

    # Load scalars
    try:
        scalar_groups = load_scalars(args.files)
    except Exception as e:
        print(f"[!] Error loading scalars: {e}")
        return

    # Validate H160
    target_h160 = args.h160.lower().strip()
    if len(target_h160) != 40:
        print(f"[!] Invalid H160 length. Should be 40 hex chars.")
        return

    # Start search
    result = search_h160(
        target_h160=target_h160,
        scalar_groups=scalar_groups,
        num_workers=args.workers,
        max_per_worker=args.attempts
    )

    # Show results
    if result and result['found']:
        print("\n" + "=" * 60)
        print("[+] SUCCESS! PRIVATE KEY FOUND")
        print("=" * 60)
        print(f"Private Key (hex): 0x{result['private']:064x}")
        print(f"Private Key (dec): {result['private']}")
        print(f"Public Key: {result['public']}")
        print(f"HASH160: {result['h160']}")
        print(f"Found after: {result['attempts']:,} attempts")
        print(f"Worker: {result['worker']}")

        # Save to file
        with open("FOUND.txt", "w") as f:
            f.write(f"PRIVATE KEY: 0x{result['private']:064x}\n")
            f.write(f"DECIMAL: {result['private']}\n")
            f.write(f"PUBLIC KEY: {result['public']}\n")
            f.write(f"H160: {result['h160']}\n")
            f.write(f"ATTEMPTS: {result['attempts']:,}\n")

        print(f"\n[+] Saved to FOUND.txt")

    else:
        print(f"\n[-] Private key not found")
        print(f"    Total attempts: {args.workers * args.attempts:,}")

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
