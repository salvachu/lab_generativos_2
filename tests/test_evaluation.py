import numpy as np
from sketchlab.evaluation import distances, geometry_metrics, prefix_preserved, polyline_points


def test_chamfer_same_geometry_independent_of_vertex_density():
    a = [np.array([[0., 0.], [100., 0.]])]
    b = [np.array([[0., 0.], [20., 0.], [100., 0.]])]
    assert distances(a, b)["chamfer"] < 1e-10


def test_no_interstroke_jump_penalty_or_phantom_segment():
    a = [np.array([[0., 0.], [1., 0.]]), np.array([[500., 500.], [501., 500.]])]
    assert geometry_metrics(a, jump_threshold=10)["intra_jump_fraction"] == 0
    p = polyline_points(a)
    assert np.all((p[:, 0] < 2) | (p[:, 0] > 499))


def test_empty_is_missing_distance_not_perfect_score():
    assert distances([], [np.array([[1., 1.]])])["chamfer"] is None
    assert geometry_metrics([])["empty"] == 1


def test_prefix_value_exactness():
    p = [np.array([[1.123456789, 3.]])]
    assert prefix_preserved(p, p + [np.array([[1., 2.]])])
    assert not prefix_preserved(p, [np.array([[1.123456788, 3.]])])
