from base64 import urlsafe_b64encode

from hashlib import blake2s
import krock32
from time import time
import secrets
import re
import struct
from typing import Optional


def generate_token(namespace: str, token_data: Optional[bytes] = None) -> str:
    """
    Generates a random, namespaced token with a time component.

    :param namespace:
    :param token_data:
    :return:
    """

    if not namespace.isalnum():
        raise ValueError("Namespace must be alphanumeric.")

    if not namespace.isupper():
        raise ValueError("Namespace must be entirely uppercase.")

    if len(namespace) > 7:
        raise ValueError("Namespace must not have more than 7 characters.")

    while len(namespace) < 7:
        namespace = f"{namespace}#"

    encoder = krock32.Encoder()

    if token_data is None:
        time_bytes = bytearray(struct.pack('d', time()))
        random_bytes = bytearray(secrets.token_bytes(10))
        concat_bytes = bytes(time_bytes + random_bytes)
    else:
        concat_bytes = token_data

    hashed_bytes = blake2s(concat_bytes, digest_size=24).digest()
    encoder.update(hashed_bytes)

    q = encoder.finalize()
    n = 10

    token = namespace + "-" + "-".join([q[i:i + n] for i in range(0, len(q), n)])

    return token


def valid_token(namespace: str, token: str) -> bool:
    token_pattern = re.compile(
        "^[A-Z,0-9]{7}-[A-Z,0-9]{10}-[A-Z,0-9]{10}-[A-Z,0-9]{10}-[A-Z,0-9]{0,10}$"
    )
    token = token.upper()

    if not token.startswith(namespace.upper()):
        return False

    if token_pattern.fullmatch(token) is None:
        return False
    else:
        return True


def make_hashable(o):
    # from https://stackoverflow.com/questions/5884066/hashing-a-dictionary
    if isinstance(o, (tuple, list)):
        return tuple((make_hashable(e) for e in o))

    if isinstance(o, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in o.items()))

    if isinstance(o, (set, frozenset)):
        return tuple(sorted(make_hashable(e) for e in o))

    return o


def hash_dict(d: dict) -> str:
    """
    Create a portable, deterministic hash of a dictionary d
    :param d: dictionary to hash
    :return: The hash of the dict
    """
    return urlsafe_b64encode(blake2s(repr(make_hashable(d)).encode()).digest()).decode()
