import numpy as np


def create_sequences(
    data,
    seq_length=20
):

    sequences = []

    for i in range(
        len(data) - seq_length
    ):

        seq = data[
            i:i + seq_length
        ]

        sequences.append(seq)

    return np.array(sequences)