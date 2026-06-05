from __future__ import annotations

from typing import Iterable, Literal, Optional

import numpy as np


_CENTRAL_INDEX_PATTERNS = {
    2: (2, 4),
    3: (1, 3, 5),
    4: (0, 2, 4, 6),
    5: (0, 1, 3, 5, 6),
    6: (1, 2, 3, 4, 5, 6),
    7: (0, 1, 2, 3, 4, 5, 6),
}


def central_dft_indices(
    num_modes: int = 3,
    fundamental_index: float = 12.5,
) -> np.ndarray:
    """Return the Table-IV central DFT indices for lossless VMD."""

    if num_modes == 1:
        return np.asarray([0.0], dtype=np.float64)
    if num_modes not in _CENTRAL_INDEX_PATTERNS:
        raise ValueError("num_modes must be 1 or an integer from 2 to 7.")
    return fundamental_index * np.asarray(
        _CENTRAL_INDEX_PATTERNS[num_modes],
        dtype=np.float64,
    )


def _inverse_square_weights(
    bins: np.ndarray,
    centers: np.ndarray,
    eps: float,
) -> np.ndarray:
    distance = bins[:, None] - centers[None, :]
    near_center = np.abs(distance) < eps
    weights = np.zeros_like(distance, dtype=np.float64)

    regular = ~near_center
    weights[regular] = np.power(distance[regular], -2)

    center_rows = near_center.any(axis=1)
    if np.any(center_rows):
        counts = near_center[center_rows].sum(axis=1, keepdims=True)
        weights[center_rows] = near_center[center_rows] / counts

    noncenter_rows = ~center_rows
    weights[noncenter_rows] /= weights[noncenter_rows].sum(axis=1, keepdims=True)
    return weights


def lossless_vmd_1d(
    signal: np.ndarray,
    num_modes: int = 3,
    center_indices: Optional[Iterable[float]] = None,
    fundamental_index: float = 12.5,
    eps: float = 1e-8,
) -> np.ndarray:
    """Closed-form lossless VMD decomposition for one real-valued signal."""

    signal = np.asarray(signal)
    if signal.ndim != 1:
        raise ValueError("signal must be one-dimensional.")

    if center_indices is None:
        centers = central_dft_indices(num_modes, fundamental_index=fundamental_index)
    else:
        centers = np.asarray(list(center_indices), dtype=np.float64)
        num_modes = int(centers.size)

    n = signal.shape[0]
    spectrum = np.fft.fft(signal)
    positive_len = n // 2 + 1
    positive_spectrum = spectrum[:positive_len]

    bins = np.arange(positive_len, dtype=np.float64)
    weights = _inverse_square_weights(bins, centers, eps=eps)
    positive_modes = positive_spectrum[:, None] * weights

    if n % 2 == 0:
        negative_modes = np.conjugate(positive_modes[1:-1][::-1])
    else:
        negative_modes = np.conjugate(positive_modes[1:][::-1])

    mode_spectra = np.concatenate([positive_modes, negative_modes], axis=0).T
    modes = np.fft.ifft(mode_spectra, axis=1)
    return np.real_if_close(modes, tol=1000).real


def decompose_iq(
    samples: np.ndarray,
    num_modes: int = 3,
    center_indices: Optional[Iterable[float]] = None,
    fundamental_index: float = 12.5,
    representation: Literal["iq", "sum", "magnitude_phase"] = "iq",
) -> np.ndarray:
    """Apply lossless VMD to IQ samples.

    Input shape can be (length, 2) or (batch, length, 2). The default IQ
    representation returns channels ordered as all I-mode components followed by
    all Q-mode components, i.e. output shape (batch, length, 2 * num_modes).
    """

    samples = np.asarray(samples)
    squeeze = False
    if samples.ndim == 2:
        samples = samples[None, ...]
        squeeze = True
    if samples.ndim != 3 or samples.shape[-1] != 2:
        raise ValueError("samples must have shape (length, 2) or (batch, length, 2).")

    outputs = []
    for sample in samples:
        i_channel = sample[:, 0]
        q_channel = sample[:, 1]

        if representation == "iq":
            i_modes = lossless_vmd_1d(
                i_channel,
                num_modes=num_modes,
                center_indices=center_indices,
                fundamental_index=fundamental_index,
            )
            q_modes = lossless_vmd_1d(
                q_channel,
                num_modes=num_modes,
                center_indices=center_indices,
                fundamental_index=fundamental_index,
            )
            decomposed = np.concatenate([i_modes, q_modes], axis=0).T
        elif representation == "sum":
            summed = i_channel + q_channel
            decomposed = lossless_vmd_1d(
                summed,
                num_modes=num_modes,
                center_indices=center_indices,
                fundamental_index=fundamental_index,
            ).T
        elif representation == "magnitude_phase":
            magnitude = np.sqrt(i_channel**2 + q_channel**2)
            phase = np.unwrap(np.arctan2(q_channel, i_channel))
            magnitude_modes = lossless_vmd_1d(
                magnitude,
                num_modes=num_modes,
                center_indices=center_indices,
                fundamental_index=fundamental_index,
            )
            phase_modes = lossless_vmd_1d(
                phase,
                num_modes=num_modes,
                center_indices=center_indices,
                fundamental_index=fundamental_index,
            )
            decomposed = np.concatenate([magnitude_modes, phase_modes], axis=0).T
        else:
            raise ValueError("representation must be 'iq', 'sum', or 'magnitude_phase'.")

        outputs.append(decomposed)

    result = np.stack(outputs, axis=0)
    return result[0] if squeeze else result


def normalize_iq_power(samples: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize IQ sequences by their average signal power."""

    samples = np.asarray(samples)
    if samples.shape[-1] != 2:
        raise ValueError("Expected IQ data with last dimension size 2.")

    power = np.sqrt(np.mean(np.sum(samples**2, axis=-1), axis=-1))
    return samples / np.maximum(power, eps)[..., None, None]
