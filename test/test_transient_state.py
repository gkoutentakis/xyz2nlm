import numpy as np
import numpy.testing as npt
import pytest
import inspect

from xyz2nlm.custom_dataclasses import TransientState

parameter_list = [(10, 0.2), (10, 0.2), (10, 0.2)]
benchmark_parameter_list = [(10, 0.2), (100, 0.2), (1000, 0.2),
                            (10, 0.5), (100, 0.5), (1000, 0.5),
                            (10, 0.8), (100, 0.8), (1000, 0.8)]
SEEDS = np.arange(10)

def get_random_states(size, sparsity, number_of_states):
    floor = lambda x: np.floor(x).astype(int)
    Nstates = ((size + 1) * (size + 2))//2

    # create random states
    state_ndarrays = np.zeros((Nstates, number_of_states), dtype=complex)
    state_objs = []

    for i in range(number_of_states):
        nnz = floor(sparsity * size)
        non_zero_ind = np.random.choice(np.arange(size), nnz, replace=False)
        non_zero_val = np.random.rand(nnz) + 1j * np.random.rand(nnz)

        state_ndarrays[non_zero_ind, i] = non_zero_val
        state_objs.append(TransientState(n=size,
                                         non_zero_ind=non_zero_ind,
                                         non_zero_val=non_zero_val))

    return state_objs, state_ndarrays


@pytest.mark.parametrize("seed", SEEDS, ids=lambda s: f"seed={s}")
@pytest.mark.parametrize("size,sparsity", parameter_list)
def test_TransientState_to_ndarray(size, sparsity, seed):
    rng = np.random.default_rng(seed)

    state_obj, state_ndarray = get_random_states(size, sparsity, 1)

    expected = state_ndarray[:, 0]
    actual = state_obj[0].as_ndarray()
    npt.assert_allclose(actual, expected, atol=1e-12)

@pytest.mark.parametrize("seed", SEEDS, ids=lambda s: f"seed={s}")
@pytest.mark.parametrize("size,sparsity", parameter_list)
def test_TransientState_arraysum(size, sparsity, seed):
    rng = np.random.default_rng(seed)

    state_objs, state_ndarrays = get_random_states(size, sparsity, 2)

    expected = np.sum(state_ndarrays, axis=1)
    sum_state_obj= state_objs[0]._array_sum(state_objs[1])
    actual = sum_state_obj.as_ndarray()
    npt.assert_allclose(actual, expected, atol=1e-12)

@pytest.mark.parametrize("size,sparsity", benchmark_parameter_list)
def test_TransientState_arraysum_benchmark(size, sparsity, benchmark):
    rng = np.random.default_rng(SEEDS[0])

    state_objs, state_ndarrays = get_random_states(size, sparsity, 2)

    expected = np.sum(state_ndarrays, axis=1)
    sum_state_obj= benchmark(state_objs[0]._array_sum, state_objs[1])
    actual = sum_state_obj.as_ndarray()
    npt.assert_allclose(actual, expected, atol=1e-12)

@pytest.mark.parametrize("seed", SEEDS, ids=lambda s: f"seed={s}")
@pytest.mark.parametrize("size,sparsity", parameter_list)
def test_TransientState_setsum(size, sparsity, seed):
    rng = np.random.default_rng(seed)

    state_objs, state_ndarrays = get_random_states(size, sparsity, 2)

    expected = np.sum(state_ndarrays, axis=1)
    sum_state_obj= state_objs[0]._set_sum(state_objs[1])
    actual = sum_state_obj.as_ndarray()
    npt.assert_allclose(actual, expected, atol=1e-12)

@pytest.mark.parametrize("size,sparsity", benchmark_parameter_list)
def test_TransientState_setsum_benchmark(size, sparsity,benchmark):
    rng = np.random.default_rng(SEEDS[0])

    state_objs, state_ndarrays = get_random_states(size, sparsity, 2)

    expected = np.sum(state_ndarrays, axis=1)
    sum_state_obj= benchmark(state_objs[0]._set_sum, state_objs[1])
    actual = sum_state_obj.as_ndarray()
    npt.assert_allclose(actual, expected, atol=1e-12)

@pytest.mark.parametrize("seed", SEEDS, ids=lambda s: f"seed={s}")
@pytest.mark.parametrize("size,sparsity", parameter_list)
def test_TransientState_sum(size, sparsity, seed):
    rng = np.random.default_rng(seed)

    state_objs, state_ndarrays = get_random_states(size, sparsity, 2)

    expected = np.sum(state_ndarrays, axis=1)
    sum_state_obj= state_objs[0] + state_objs[1]
    actual = sum_state_obj.as_ndarray()
    npt.assert_allclose(actual, expected, atol=1e-12)

@pytest.mark.parametrize("seed", SEEDS, ids=lambda s: f"seed={s}")
@pytest.mark.parametrize("size,sparsity", parameter_list)
def test_TransientState_scaling(size, sparsity, seed):
    rng = np.random.default_rng(seed)

    state_obj, state_ndarray = get_random_states(size, sparsity, 1)

    scale = np.random.rand() + 1j * np.random.rand()
    expected = scale * state_ndarray[:, 0]
    scaled_state_obj= state_obj[0].scale(scale)
    actual = scaled_state_obj.as_ndarray()
    npt.assert_allclose(actual, expected, atol=1e-12)

@pytest.mark.parametrize("seed", SEEDS, ids=lambda s: f"seed={s}")
@pytest.mark.parametrize("size,sparsity", parameter_list)
def test_TransientState_as_ndarray(size, sparsity, seed):
    rng = np.random.default_rng(seed)

    state_obj, state_ndarray = get_random_states(size, sparsity, 1)

    state_obj_from_ndarray = TransientState.from_ndarray(size, state_ndarray)
    difference = state_obj[0] + state_obj_from_ndarray.scale(-1.0)
    npt.assert_allclose(difference.as_ndarray(), 0, atol=1e-12)
