"""Calibrate receiver context tables from observed three-copy updates."""
from collections import deque
from math import exp, log, prod
from operator import index


def fit_context(runs, bits, prior=None):
    """Estimate drop probabilities for each offer position and recent packet history."""
    nodes = (2 << bits) - 1
    attempts = [[0]*nodes for _ in range(3)]
    misses = [[0]*nodes for _ in range(3)]
    for run in runs:
        history = depth = 0
        for mask in run:
            for ordinal in range(3):
                lost = not (mask & (1 << ordinal))
                for length in range(depth + 1):
                    start = (1 << length) - 1
                    node = start + (history & start)
                    attempts[ordinal][node] += 1
                    misses[ordinal][node] += lost
                history = ((history << 1) | lost) & ((1 << bits) - 1)
                depth = min(bits, depth + 1)
    probabilities = []
    for ordinal in range(3):
        row = []
        for node, (lost, total) in enumerate(zip(misses[ordinal], attempts[ordinal])):
            depth = (node + 1).bit_length() - 1
            # Back off by dropping the oldest bit, preserving the newest outcomes.
            shorter = (1 << max(0, depth-1)) - 1
            history = node + 1
            backoff = row[shorter + (history & shorter)] if depth else .5
            reference = prior[ordinal][node] if prior is not None else backoff
            prefix = (1 << ordinal) - 1
            if depth > ordinal and (history & prefix) != prefix:
                # A copy already landed. Borrow a local observation conditional on
                # this update's prefix when estimating the remaining copy losses.
                reference = row[prefix + (history & prefix)]
            row.append((lost + reference + backoff) / (total + 2))
        probabilities.append(row)
    return probabilities


def _observations(runs):
    # Materialize once so generators and caller-owned mutable buffers are safe.
    data = [tuple(map(index, run)) for run in runs]
    counts = [0]*8
    for run in data:
        for mask in run:
            if not 0 <= mask < 8:
                raise ValueError("expected a three-copy reception mask in [0, 7]")
            counts[mask] += 1
    if not any(counts):
        raise ValueError("at least one witnessed update is required")
    return data, counts


def _excess_fraction(counts):
    """Fraction of complete misses above the independent-copy expectation."""
    if not counts[0]:
        return 0.0
    total = sum(counts)
    marginal = [sum(n for mask, n in enumerate(counts) if not (mask & (1 << j))) / total
                for j in range(3)]
    return max(0.0, 1 - prod(marginal) * total / counts[0])


def _triplet_paths(bits):
    """For each starting node and possible mask, follow three receiver history updates."""
    leaves = (1 << bits) - 1
    paths = []
    for start in range(2*leaves + 1):
        row = []
        for mask in range(8):
            node = start
            path = []
            for j in range(3):
                path.append(node)
                loss = int(not (mask & (1 << j)))
                if node < leaves:
                    node = 2*node + 1 + loss
                else:
                    node = leaves + ((2*(node-leaves) + loss) & leaves)
            row.append((*path, node))
        paths.append(row)
    return paths


def _expected_masks(table, paths, lengths):
    """Finite-run mask frequencies under one table, starting from empty history at every gap."""
    ends = [0]*(max(lengths)+1)
    for length in lengths:
        ends[length] += 1
    transitions = {}
    state = {0: 1.0}
    counts = [0.0]*8
    remaining = len(lengths) - ends[0]
    for tick in range(1, len(ends)):
        next_state = {}
        for start, mass in state.items():
            if start not in transitions:
                row = []
                for mask, (a, b, c, end) in enumerate(paths[start]):
                    p, q, r = table[0][a], table[1][b], table[2][c]
                    prob = ((1 - p if mask & 1 else p) * (1 - q if mask & 2 else q)
                            * (1 - r if mask & 4 else r))
                    row.append((prob, end))
                transitions[start] = row
            for mask, (prob, end) in enumerate(transitions[start]):
                emitted = mass*prob
                counts[mask] += remaining*emitted
                next_state[end] = next_state.get(end, 0.0) + emitted
        remaining -= ends[tick]
        state = next_state
    total = sum(lengths)
    return [n/total for n in counts]


class TripletLossCalibrator:
    """One session on one directed link. Each observe() call supplies a pilot block.

    Each inner run contains consecutive three-bit reception masks, earliest copy
    in the low bit. Break runs at counter gaps. Only supply independently
    witnessed offers, including complete misses (mask 0).
    """
    def __init__(self, reference_runs, history_bits=5):
        self.history_bits = index(history_bits)
        if not 1 <= self.history_bits <= 8:
            raise ValueError("history_bits must be in [1, 8]")
        data, counts = _observations(reference_runs)
        self._reference = fit_context(data, self.history_bits)
        self._reference_excess = _excess_fraction(counts)
        self._paths = _triplet_paths(self.history_bits)
        self.reset_session()

    def reset_session(self):
        """Start a new receiver session while retaining the reference capture."""
        self._counts = [0]*8
        self._pilots = deque(maxlen=8)
        self._scores = [0.0]*51
        self.reference_weight = 0.0

    def observe(self, pilot_runs):
        """Return probabilities [3, nodes] for the next offers on this link."""
        data, counts = _observations(pilot_runs)
        self._pilots.append(data)
        self._counts = [old + new for old, new in zip(self._counts, counts)]
        excess = _excess_fraction(self._counts)
        cap = min(1.0, excess / self._reference_excess) if self._reference_excess else 0.0
        recent = [run for block in self._pilots for run in block]
        local = fit_context(recent, self.history_bits)
        if not self._reference_excess:
            return local
        borrowed = fit_context(recent, self.history_bits, self._reference)
        lengths = list(map(len, recent))
        local_masks = _expected_masks(local, self._paths, lengths)
        borrowed_masks = _expected_masks(borrowed, self._paths, lengths)
        # Score each new pilot observation once. The two end-point distributions
        # are exact. Their mixture is a calibration score for choosing a table.
        intervals = len(self._scores) - 1
        for i in range(intervals + 1):
            fraction = i / intervals
            for n, a, b in zip(counts, local_masks, borrowed_masks):
                if n:
                    self._scores[i] += n * log(max(1e-30, (1 - fraction) * a + fraction * b))
        position = cap * intervals
        left = int(position)
        score = self._scores[left]
        if left < intervals:
            score += (position - left) * (self._scores[left + 1] - score)
        best = max(score, max(self._scores[:left + 1]))
        weight = cap * exp(score - best)
        self.reference_weight = weight
        return [[(1 - weight) * a + weight * b for a, b in zip(local_row, borrowed_row)]
                for local_row, borrowed_row in zip(local, borrowed)]
