# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU teaching demos; these do not implement or benchmark vLLM."""

import argparse
import hashlib
import json
import math
from dataclasses import dataclass


def project(vector: list[float], weight: list[list[float]]) -> list[float]:
    return [
        sum(value * row[column] for value, row in zip(vector, weight))
        for column in range(len(weight[0]))
    ]


def attend(
    query: list[float], keys: list[list[float]], values: list[list[float]]
) -> list[float]:
    scores = [
        sum(q * k for q, k in zip(query, key)) / math.sqrt(len(query)) for key in keys
    ]
    weights = [math.exp(score - max(scores)) for score in scores]
    denominator = sum(weights)
    return [
        sum(weight * value[column] for weight, value in zip(weights, values))
        / denominator
        for column in range(len(values[0]))
    ]


def attention_demo() -> None:
    inputs = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, -0.5]]
    wq = [[1.0, 0.5], [-0.5, 1.0]]
    wk = [[0.5, 1.0], [1.0, -0.5]]
    wv = [[1.0, -1.0], [0.5, 0.5]]
    cache_k: list[list[float]] = []
    cache_v: list[list[float]] = []
    for position, vector in enumerate(inputs):
        query = project(vector, wq)
        prefix = inputs[: position + 1]
        reference = attend(
            query,
            [project(item, wk) for item in prefix],
            [project(item, wv) for item in prefix],
        )
        cache_k.append(project(vector, wk))
        cache_v.append(project(vector, wv))
        cached = attend(query, cache_k, cache_v)
        assert reference == cached
        print(f"position={position}, output={cached}, cached_matches_full=True")
    count = len(inputs)
    print(
        "K/V projection rows per matrix: "
        f"recompute={count * (count + 1) // 2}, cache={count}"
    )
    print("One attention layer, fixed inputs, no RoPE/FFN/GPU timing.")


def prefix_hashes(tokens: list[int], namespace: str, block_size: int) -> list[str]:
    parent = ""
    hashes = []
    for offset in range(0, len(tokens) - block_size + 1, block_size):
        payload = json.dumps(
            [namespace, parent, tokens[offset : offset + block_size]],
            separators=(",", ":"),
        ).encode()
        parent = hashlib.sha256(payload).hexdigest()
        hashes.append(parent)
    return hashes


def prefix_demo() -> None:
    first = [1, 2, 3, 4, 5, 6]
    changed = [1, 2, 9, 4, 5, 6]
    original = prefix_hashes(first, "model-A", 2)
    different_prefix = prefix_hashes(changed, "model-A", 2)
    different_model = prefix_hashes(first, "model-B", 2)
    same_blocks = [a == b for a, b in zip(original, different_prefix)]
    assert same_blocks == [True, False, False]
    assert all(a != b for a, b in zip(original, different_model))
    print(f"Changed middle block, matching hashes: {same_blocks}")
    print("Identical last-block tokens do not restore the prefix identity.")
    print("Changing the model namespace invalidates every block in this example.")


@dataclass
class ToyRequest:
    name: str
    prefill_left: int
    decode_steps_left: int
    arrival: int = 0

    @property
    def done(self) -> bool:
        return self.prefill_left == 0 and self.decode_steps_left == 0


def schedule_demo() -> None:
    requests = [
        ToyRequest("A", 0, 3),
        ToyRequest("B", 0, 2),
        ToyRequest("C", 10, 2),
        ToyRequest("D", 2, 1, arrival=1),
    ]
    budget = 8
    step = 0
    while not all(request.done for request in requests):
        ready = [
            request
            for request in requests
            if request.arrival <= step and not request.done
        ]
        decodes = [request for request in ready if request.prefill_left == 0]
        prefills = [request for request in ready if request.prefill_left > 0]
        remaining = budget
        scheduled: dict[str, tuple[str, int]] = {}
        for request in decodes:
            if remaining == 0:
                break
            request.decode_steps_left -= 1
            remaining -= 1
            scheduled[request.name] = ("decode", 1)
        for request in prefills:
            count = min(request.prefill_left, remaining)
            if count == 0:
                break
            request.prefill_left -= count
            remaining -= count
            scheduled[request.name] = ("prefill", count)
        assert 0 <= remaining <= budget
        print(f"step={step + 1}, scheduled={scheduled}, unused={remaining}")
        step += 1
    print("decode_steps_left counts future forwards, not total output tokens.")
    print("Toy policy: no KV limits, preemption, remote loads or async execution.")


def capacity_demo() -> None:
    per_token = 2 * 32 * 8 * 128 * 2
    lengths = [17, 31]
    block_size = 16
    blocks = sum(math.ceil(length / block_size) for length in lengths)
    transfer_ms = (1024**3) / (400 * 10**9 / 8) * 1000
    assert per_token == 128 * 1024
    assert per_token * 8192 == 1024**3
    assert blocks == 4
    print(f"KV per token: {per_token / 1024:.0f} KiB")
    print(f"KV for 8192 tokens: {per_token * 8192 / 1024**3:.0f} GiB")
    print(f"Lengths={lengths}, blocks={blocks}, reserved_slots={blocks * block_size}")
    print(f"1 GiB / 400 Gbit/s ideal transfer lower bound: {transfer_ms:.2f} ms")


def main() -> None:
    demos = {
        "attention": attention_demo,
        "prefix": prefix_demo,
        "schedule": schedule_demo,
        "capacity": capacity_demo,
    }
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("demo", choices=["all", *demos], nargs="?", default="all")
    args = parser.parse_args()
    selected = demos if args.demo == "all" else {args.demo: demos[args.demo]}
    for name, demo in selected.items():
        print(f"\n[{name}]")
        demo()


if __name__ == "__main__":
    main()
