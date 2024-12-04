import numpy as np


def pad_labels(labels, max_length, pad_token_label_id):
    processed_labels = []
    for label_seq in labels:
        padded_seq = label_seq + [pad_token_label_id] * (max_length - len(label_seq))
        processed_labels.append(padded_seq)
    return np.array(processed_labels)