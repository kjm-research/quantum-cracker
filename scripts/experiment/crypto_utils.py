"""Crypto utilities for wallet generation and address feature extraction.

Uses ecdsa for secp256k1 key generation and pycryptodome for hashing.
"""

import hashlib
import struct

from ecdsa import SECP256k1, SigningKey
from Crypto.Hash import keccak as crypto_keccak


def generate_keypair():
    """Generate a secp256k1 keypair.

    Returns:
        dict with keys: private_key_hex, public_key_hex, public_key_x, public_key_y
    """
    sk = SigningKey.generate(curve=SECP256k1)
    vk = sk.get_verifying_key()

    priv_hex = sk.to_string().hex()
    pub_bytes = vk.to_string()  # 64 bytes: x (32) + y (32)
    pub_hex = pub_bytes.hex()

    x = int.from_bytes(pub_bytes[:32], "big")
    y = int.from_bytes(pub_bytes[32:], "big")

    return {
        "private_key_hex": priv_hex,
        "public_key_hex": pub_hex,
        "public_key_x": f"{x:064x}",
        "public_key_y": f"{y:064x}",
        "public_key_x_int": x,
        "public_key_y_int": y,
    }


def eth_address_from_pubkey(pub_hex):
    """Derive Ethereum address from uncompressed public key (64-byte hex).

    ETH address = last 20 bytes of keccak256(public_key_bytes).
    """
    pub_bytes = bytes.fromhex(pub_hex)
    h = crypto_keccak.new(digest_bits=256)
    h.update(pub_bytes)
    addr_bytes = h.digest()[-20:]
    return "0x" + addr_bytes.hex()


def btc_address_from_pubkey(pub_hex):
    """Derive Bitcoin address (P2PKH) from uncompressed public key.

    1. Prepend 0x04 to the 64-byte public key
    2. SHA256 -> RIPEMD160 (Hash160)
    3. Prepend version byte 0x00
    4. SHA256(SHA256(versioned)) -> first 4 bytes = checksum
    5. Base58Check encode
    """
    pub_bytes = b"\x04" + bytes.fromhex(pub_hex)

    sha = hashlib.sha256(pub_bytes).digest()
    ripe = hashlib.new("ripemd160", sha).digest()

    versioned = b"\x00" + ripe
    checksum = hashlib.sha256(hashlib.sha256(versioned).digest()).digest()[:4]
    address_bytes = versioned + checksum

    return base58_encode(address_bytes)


def base58_encode(data):
    """Base58 encoding (Bitcoin alphabet)."""
    alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
    n = int.from_bytes(data, "big")
    result = ""
    while n > 0:
        n, r = divmod(n, 58)
        result = alphabet[r] + result

    # Preserve leading zero bytes
    for byte in data:
        if byte == 0:
            result = "1" + result
        else:
            break

    return result


def address_features(address_str):
    """Extract numerical features from an address string.

    Returns dict of features for correlation analysis.
    """
    # Strip 0x prefix if present
    clean = address_str.lower().replace("0x", "")
    addr_bytes = bytes.fromhex(clean) if all(c in "0123456789abcdef" for c in clean) else clean.encode()

    features = {}

    # Byte frequency distribution
    byte_freq = [0] * 256
    for b in addr_bytes:
        byte_freq[b] += 1
    features["byte_freq"] = byte_freq

    # Nibble frequency distribution (hex chars)
    nibble_freq = [0] * 16
    for c in clean:
        if c in "0123456789abcdef":
            nibble_freq[int(c, 16)] += 1
    features["nibble_freq"] = nibble_freq

    # Longest run of repeated chars
    max_run = 1
    current_run = 1
    for i in range(1, len(clean)):
        if clean[i] == clean[i - 1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
    features["longest_run"] = max_run

    # Leading zero count
    leading_zeros = 0
    for c in clean:
        if c == "0":
            leading_zeros += 1
        else:
            break
    features["leading_zeros"] = leading_zeros

    # Trailing zero count
    trailing_zeros = 0
    for c in reversed(clean):
        if c == "0":
            trailing_zeros += 1
        else:
            break
    features["trailing_zeros"] = trailing_zeros

    # Shannon entropy of the hex string
    import math
    char_counts = {}
    for c in clean:
        char_counts[c] = char_counts.get(c, 0) + 1
    total = len(clean)
    entropy = -sum((count / total) * math.log2(count / total) for count in char_counts.values())
    features["entropy"] = entropy

    # Numeric sum of all hex digits
    features["digit_sum"] = sum(int(c, 16) for c in clean if c in "0123456789abcdef")

    # Number of unique hex chars used
    features["unique_chars"] = len(set(clean))

    return features


def generate_wallet():
    """Generate a complete wallet with all addresses and features.

    Returns dict with private key, public key, addresses, and features.
    """
    kp = generate_keypair()

    eth_addr = eth_address_from_pubkey(kp["public_key_hex"])
    btc_addr = btc_address_from_pubkey(kp["public_key_hex"])

    eth_features = address_features(eth_addr)
    # BTC address is base58, extract features differently
    btc_features = {
        "length": len(btc_addr),
        "leading_ones": sum(1 for c in btc_addr if c == "1") - sum(1 for i, c in enumerate(btc_addr) if c == "1" and (i == 0 or btc_addr[i-1] == "1")),
    }

    return {
        "private_key_hex": kp["private_key_hex"],
        "public_key_hex": kp["public_key_hex"],
        "public_key_x": kp["public_key_x"],
        "public_key_y": kp["public_key_y"],
        "eth_address": eth_addr,
        "btc_address": btc_addr,
        "eth_features": eth_features,
    }


if __name__ == "__main__":
    print("Generating test wallet...")
    w = generate_wallet()
    print(f"  Private key: {w['private_key_hex']}")
    print(f"  Public key:  {w['public_key_hex'][:32]}...")
    print(f"  ETH address: {w['eth_address']}")
    print(f"  BTC address: {w['btc_address']}")
    print(f"  ETH entropy: {w['eth_features']['entropy']:.4f}")
    print(f"  ETH nibble freq: {w['eth_features']['nibble_freq']}")
