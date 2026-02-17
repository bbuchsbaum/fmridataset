"""Cross-language parity tests for SamplingFrame (rpy2)."""

import numpy as np
import pytest

from fmridataset import SamplingFrame

pytestmark = pytest.mark.parity


@pytest.fixture()
def sf_params():
    return {"blocklens": [100, 120, 110], "TR": 2.0}


def test_n_timepoints(sf_params, r_fmrihrf) -> None:
    """Python n_timepoints matches R sum(blocklens)."""
    from rpy2.robjects import IntVector, FloatVector

    r_sf = r_fmrihrf.sampling_frame(
        blocklens=IntVector(sf_params["blocklens"]),
        TR=FloatVector([sf_params["TR"]]),
    )

    py_sf = SamplingFrame.create(**sf_params)

    r_n = int(sum(r_fmrihrf.blocklens(r_sf)))
    assert py_sf.n_timepoints == r_n


def test_blockids(sf_params, r_fmrihrf) -> None:
    """Python blockids matches R blockids."""
    from rpy2.robjects import IntVector, FloatVector

    r_sf = r_fmrihrf.sampling_frame(
        blocklens=IntVector(sf_params["blocklens"]),
        TR=FloatVector([sf_params["TR"]]),
    )

    py_sf = SamplingFrame.create(**sf_params)

    r_ids = np.array(r_fmrihrf.blockids(r_sf), dtype=np.intp)
    np.testing.assert_array_equal(py_sf.blockids, r_ids)


def test_total_duration(sf_params, r_fmrihrf) -> None:
    """Python total_duration matches R sum(blocklens * TR)."""
    py_sf = SamplingFrame.create(**sf_params)
    expected = sum(b * sf_params["TR"] for b in sf_params["blocklens"])
    assert py_sf.total_duration == expected
