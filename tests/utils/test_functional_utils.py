import numpy as np
from elephas.utils import functional_utils


def test_add_params():
    p1 = [np.ones((5, 5)) for _ in range(10)]
    p2 = [np.ones((5, 5)) for _ in range(10)]

    res = functional_utils.add_params(p1, p2)
    assert len(res) == 10
    for i in range(5):
        for j in range(5):
            assert res[0][i, j] == 2


def test_subtract_params():
    p1 = [np.ones((5, 5)) for _ in range(10)]
    p2 = [np.ones((5, 5)) for _ in range(10)]

    res = functional_utils.subtract_params(p1, p2)

    assert len(res) == 10
    for i in range(5):
        for j in range(5):
            assert res[0][i, j] == 0


def test_get_neutral():
    x = [np.ones((3, 4))]
    res = functional_utils.get_neutral(x)
    assert res[0].shape == x[0].shape
    assert res[0][0, 0] == 0


def test_divide_by():
    x = [np.ones((3, 4))]
    res = functional_utils.divide_by(x, num_workers=10)
    assert res[0].shape == x[0].shape
    assert res[0][0, 0] == 0.1


def test_average_and_subtract_matches_iterated_subtract():
    rng = np.random.default_rng(0)
    base = [rng.standard_normal((4, 3)).astype(np.float32) for _ in range(3)]
    deltas = [
        [rng.standard_normal((4, 3)).astype(np.float32) for _ in range(3)]
        for _ in range(5)
    ]

    expected = [b.copy() for b in base]
    n = len(deltas)
    for d in deltas:
        weighted = functional_utils.divide_by(d, n)
        expected = functional_utils.subtract_params(expected, weighted)

    actual = functional_utils.average_and_subtract(base, deltas)

    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        assert np.allclose(a, e, atol=1e-5)


def test_average_and_subtract_no_deltas_returns_copy_of_base():
    base = [np.ones((2, 2), dtype=np.float32)]
    out = functional_utils.average_and_subtract(base, [])
    assert np.array_equal(out[0], base[0])
    out[0][0, 0] = 99.0
    assert base[0][0, 0] == 1.0  # base must not be mutated


def test_average_and_subtract_does_not_mutate_inputs():
    base = [np.ones((2, 2), dtype=np.float32)]
    deltas = [
        [np.full((2, 2), 0.5, dtype=np.float32)],
        [np.full((2, 2), 0.5, dtype=np.float32)],
    ]
    base_orig = [b.copy() for b in base]
    deltas_orig = [[d.copy() for d in dl] for dl in deltas]

    functional_utils.average_and_subtract(base, deltas)

    for a, b in zip(base, base_orig):
        assert np.array_equal(a, b)
    for dl, dl_orig in zip(deltas, deltas_orig):
        for d, d_orig in zip(dl, dl_orig):
            assert np.array_equal(d, d_orig)
