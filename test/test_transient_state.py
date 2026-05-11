import numpy as np
import numpy.testing as npt
import pytest

from xyz2nlm.custom_dataclasses import TransientStateNP, TransientStateQI


STATE_CLASSES = [
    pytest.param(TransientStateNP, id="np"),
    pytest.param(TransientStateQI, id="qi"),
]

parameter_list = [(10, 0.2), (10, 0.5), (10, 0.8)]
benchmark_parameter_list = [
    (10, 0.2), (100, 0.2), (1000, 0.2),
    (10, 0.5), (100, 0.5), (1000, 0.5),
    (10, 0.8), (100, 0.8), (1000, 0.8),
]
SEEDS = np.arange(10)


def state_as_complex_array(state):
    if isinstance(state, TransientStateQI):
        return state.as_complex_ndarray()
    return state.as_ndarray()


def assert_state_matches_array(state, expected, atol=1e-12):
    npt.assert_allclose(state_as_complex_array(state), expected, atol=atol)


def get_random_states(state_class, size, sparsity, number_of_states, seed):
    rng = np.random.default_rng(seed)
    floor = lambda x: np.floor(x).astype(int)
    nstates = ((size + 1) * (size + 2)) // 2

    state_ndarrays = np.zeros((nstates, number_of_states), dtype=complex)
    state_objs = []

    for i in range(number_of_states):
        nnz = max(1, floor(sparsity * size))
        non_zero_ind = rng.choice(np.arange(nstates), nnz, replace=False)
        non_zero_val = rng.random(nnz) + 1j * rng.random(nnz)

        state_ndarrays[non_zero_ind, i] = non_zero_val
        state_objs.append(
            state_class(n=size, non_zero_ind=non_zero_ind, non_zero_val=non_zero_val)
        )

    return state_objs, state_ndarrays


@pytest.mark.parametrize("state_class", STATE_CLASSES)
def test_TransientState_storage(state_class):
    state = state_class(
        n=3,
        non_zero_ind=np.array([0, 2, 5]),
        non_zero_val=np.array([1 + 0j, 2 - 1j, -3j]),
    )

    assert state.n == 3
    npt.assert_array_equal(state.non_zero_ind, np.array([0, 2, 5]))
    npt.assert_allclose(state_as_complex_array(state)[[0, 2, 5]], [1, 2 - 1j, -3j])


@pytest.mark.parametrize("state_class", STATE_CLASSES)
@pytest.mark.parametrize("seed", SEEDS, ids=lambda s: f"seed={s}")
@pytest.mark.parametrize("size,sparsity", parameter_list)
def test_TransientState_to_ndarray(state_class, size, sparsity, seed):
    state_obj, state_ndarray = get_random_states(state_class, size, sparsity, 1, seed)
    assert_state_matches_array(state_obj[0], state_ndarray[:, 0])


@pytest.mark.parametrize("state_class", STATE_CLASSES)
@pytest.mark.parametrize("seed", SEEDS, ids=lambda s: f"seed={s}")
@pytest.mark.parametrize("size,sparsity", parameter_list)
def test_TransientState_arraysum(state_class, size, sparsity, seed):
    state_objs, state_ndarrays = get_random_states(state_class, size, sparsity, 2, seed)

    expected = np.sum(state_ndarrays, axis=1)
    sum_state_obj = state_objs[0]._array_sum(state_objs[1])
    assert_state_matches_array(sum_state_obj, expected)


@pytest.mark.parametrize("state_class", STATE_CLASSES)
@pytest.mark.parametrize("seed", SEEDS, ids=lambda s: f"seed={s}")
@pytest.mark.parametrize("size,sparsity", parameter_list)
def test_TransientState_setsum(state_class, size, sparsity, seed):
    state_objs, state_ndarrays = get_random_states(state_class, size, sparsity, 2, seed)

    expected = np.sum(state_ndarrays, axis=1)
    sum_state_obj = state_objs[0]._set_sum(state_objs[1])
    assert_state_matches_array(sum_state_obj, expected)


@pytest.mark.parametrize("state_class", STATE_CLASSES)
@pytest.mark.parametrize("seed", SEEDS, ids=lambda s: f"seed={s}")
@pytest.mark.parametrize("size,sparsity", parameter_list)
def test_TransientState_sum(state_class, size, sparsity, seed):
    state_objs, state_ndarrays = get_random_states(state_class, size, sparsity, 2, seed)

    expected = np.sum(state_ndarrays, axis=1)
    sum_state_obj = state_objs[0] + state_objs[1]
    assert_state_matches_array(sum_state_obj, expected)


@pytest.mark.parametrize("state_class", STATE_CLASSES)
@pytest.mark.parametrize("seed", SEEDS, ids=lambda s: f"seed={s}")
@pytest.mark.parametrize("size,sparsity", parameter_list)
def test_TransientState_scaling(state_class, size, sparsity, seed):
    state_obj, state_ndarray = get_random_states(state_class, size, sparsity, 1, seed)
    rng = np.random.default_rng(seed + 100)

    scale = rng.random() + 1j * rng.random()
    expected = scale * state_ndarray[:, 0]
    scaled_state_obj = state_obj[0].scale(scale)
    assert_state_matches_array(scaled_state_obj, expected)


@pytest.mark.parametrize("state_class", STATE_CLASSES)
@pytest.mark.parametrize("seed", SEEDS, ids=lambda s: f"seed={s}")
@pytest.mark.parametrize("size,sparsity", parameter_list)
def test_TransientState_from_ndarray(state_class, size, sparsity, seed):
    state_obj, state_ndarray = get_random_states(state_class, size, sparsity, 1, seed)

    state_obj_from_ndarray = state_class.from_ndarray(size, state_ndarray[:, 0])
    difference = state_obj[0] + state_obj_from_ndarray.scale(-1.0)
    assert_state_matches_array(difference, np.zeros_like(state_ndarray[:, 0]))


@pytest.mark.parametrize("state_class", STATE_CLASSES)
def test_TransientState_merges_overlapping_indices_and_removes_zero_sum(state_class):
    state_a = state_class(
        n=3,
        non_zero_ind=np.array([0, 2, 4]),
        non_zero_val=np.array([1, 2, 3], dtype=complex),
    )
    state_b = state_class(
        n=3,
        non_zero_ind=np.array([2, 3, 4]),
        non_zero_val=np.array([-2, 5, 1], dtype=complex),
    )

    actual = state_a + state_b
    expected = np.zeros((10,), dtype=complex)
    expected[[0, 3, 4]] = [1, 5, 4]

    assert 2 not in actual.non_zero_ind
    assert_state_matches_array(actual, expected)


@pytest.mark.parametrize("state_class", STATE_CLASSES)
def test_TransientState_from_ndarray_removes_zero_coefficients(state_class):
    dense = np.zeros((10,), dtype=complex)
    dense[[1, 4]] = [2 - 1j, -3j]

    state = state_class.from_ndarray(3, dense)

    npt.assert_array_equal(state.non_zero_ind, np.array([1, 4]))
    assert_state_matches_array(state, dense)


@pytest.mark.parametrize("state_class", STATE_CLASSES)
def test_TransientState_as_complex_ndarray(state_class):
    state = state_class(
        n=3,
        non_zero_ind=np.array([0, 5]),
        non_zero_val=np.array([1 + 2j, -3j]),
    )

    actual = state.as_complex_ndarray()

    assert actual.dtype == complex
    expected = np.zeros((10,), dtype=complex)
    expected[[0, 5]] = [1 + 2j, -3j]
    npt.assert_allclose(actual, expected)


@pytest.mark.parametrize("size,sparsity", benchmark_parameter_list)
def test_TransientState_arraysum_benchmark(size, sparsity, benchmark):
    state_objs, state_ndarrays = get_random_states(
        TransientStateNP, size, sparsity, 2, SEEDS[0]
    )

    expected = np.sum(state_ndarrays, axis=1)
    sum_state_obj = benchmark(state_objs[0]._array_sum, state_objs[1])
    assert_state_matches_array(sum_state_obj, expected)


@pytest.mark.parametrize("size,sparsity", benchmark_parameter_list)
def test_TransientState_setsum_benchmark(size, sparsity, benchmark):
    state_objs, state_ndarrays = get_random_states(
        TransientStateNP, size, sparsity, 2, SEEDS[0]
    )

    expected = np.sum(state_ndarrays, axis=1)
    sum_state_obj = benchmark(state_objs[0]._set_sum, state_objs[1])
    assert_state_matches_array(sum_state_obj, expected)
