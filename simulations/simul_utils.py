import os


def get_simul_name(simul_case, Ns, T, rs, rt):
    simul_name = "_".join(
        [simul_case , "Ns" + str(Ns) , "T" + str(T) , "rs" + str(rs)[2:] , "rt" + str(rt)[2:]]
    )
    return simul_name


def get_config_path(simul_case, simul_name):
    SINGLE_CFG = True
    if not os.path.exists("./simulation_configs"):
        raise ValueError("Directory to read simulation configurations does not exist.")
    else:
        if SINGLE_CFG:
            config_path = os.path.join("./simulation_configs/", simul_case + ".yaml")
        else:
            config_path = os.path.join("./simulation_configs/", simul_name + ".yaml")
    return config_path


def get_case_path(simul_case):
    if not os.path.exists("./simulation_cases"):
        raise ValueError("Directory to read simulation cases does not exist.")
    else:
        case_path = os.path.join("./simulation_cases/", simul_case + ".yaml")
    return case_path


def add_common_arguments(parser):
    parser.add_argument(
        "--case", "-c", required=True, type=str, help="Simulation case."
    )
    parser.add_argument(
        "--num-subject", "-s", required=True, type=int, help="Number of subjects."
    )
    parser.add_argument(
        "--time-horizon", "-t", required=True, type=int, help="Number of time steps."
    )
    parser.add_argument(
        "--r-subject",
        required=True,
        type=float,
        help="Probability of a node changing clusters along subject axis.",
    )
    parser.add_argument(
        "--r-time",
        required=True,
        type=float,
        help="Probability of a node changing clusters along time axis",
    )
    ## parser.add_argument(
    ##     "--num-vertices",
    ##     "-n",
    ##     required=True,
    ##     type=int,
    ##     help="Total number of vertices in the network.",
    ## )
    ## parser.add_argument(
    ##     "--num-cluster",
    ##     "-K",
    ##     required=True,
    ##     type=int,
    ##     help="Number of clusters for dcbm and K-means.",
    ## )
    ## parser.add_argument(
    ##     "--p-in",
    ##     required=True,
    ##     type=float,
    ##     nargs="2",
    ##     help="In-cluster density parameters, p_in=(p_in[0], p_in^[1]).",
    ## )
    ## parser.add_argument(
    ##     "--p-out",
    ##     required=True,
    ##     type=float,
    ##     help="Between-cluster denisty parameter p_out.",
    ## )
    return parser
