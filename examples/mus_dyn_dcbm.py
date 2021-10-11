# One easy example for MuSPCES.
import sys

sys.path.append("../")

from mudcod.dcbm import MuSDynamicDCBM  # noqa: E402


def print_shape(adj_ms_series, z_ms):
    print(
        f"Shape of the multi subject adjacency time series is : {adj_ms3_series.shape}",
    )
    print("(number of subjects, time-horizon, n, n)")
    print(
        f"Shape of the multi subject community label time series is : {z_ms3_true.shape}",
    )
    print("(number of subjects, time-horizon, n)")


# Multi-subject Dynamic DCBM
model_dcbm = MuSDynamicDCBM(
    n=100,  # Total number of vertices
    k=2,  # Model order, i.e. number of class labels.
    p_in=(0.2, 0.25),  # In class connectivity parameter
    p_out=0.1,  # Out class connectivity parameter
    time_horizon=4,  # Total number of time steps
    r_time=0.2,  # Probability of changing class labels in the next time step
    num_subjects=8,  # Total number of subjects
    r_subject=0.2,  # Probability of changing class labels while evolving
)

# scenario 3 (SSoS): strong signal sharing among subjects.
adj_ms3_series, z_ms3_true = model_dcbm.simulate_ms_dynamic_dcbm(scenario=3)
print_shape(adj_ms3_series, z_ms3_true)

# scenario 1 (SSoT): signal sharing over time, subjects evolve independently.
adj_ms1_series, z_ms1_true = model_dcbm.simulate_ms_dynamic_dcbm(scenario=1)
print_shape(adj_ms1_series, z_ms1_true)

# You can similarly generate single-subject dynamic DCBM networks and static DCBM networks.
