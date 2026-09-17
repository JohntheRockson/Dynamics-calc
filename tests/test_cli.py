"""CLI wiring for SimLab scenarios (no plant/controller rewrites)."""

from attitude_sim.sim import build_parser, main, make_scenario_config


def test_parser_defaults_to_slew():
    args = build_parser().parse_args([])
    assert args.scenario == "slew"
    assert args.controller == "pid"
    assert args.estimator == "mekf"


def test_parser_detumble_and_flags():
    args = build_parser().parse_args(
        ["--scenario", "detumble", "--controller", "lqr", "--no-plot", "--no-gif"]
    )
    assert args.scenario == "detumble"
    assert args.controller == "lqr"
    assert args.no_plot
    assert args.no_gif


def test_make_scenario_config_stems():
    slew = make_scenario_config("slew", plot=False, gif=False)
    det = make_scenario_config("detumble", plot=False, gif=False)
    assert slew.artifact_stem == "slew"
    assert det.artifact_stem == "detumble"
    assert abs(det.omega0).max() > 0.2


def test_main_both_scenarios_smoke():
    assert main(["--scenario", "slew", "--t-final", "0.05", "--no-plot", "--no-gif"]) == 0
    assert main(["--scenario", "detumble", "--t-final", "0.05", "--no-plot", "--no-gif"]) == 0
