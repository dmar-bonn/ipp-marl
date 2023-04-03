import time
import optuna

import constants
from logger import setup_logger
from marl_framework.missions.mission_factories import MissionFactory
from params import load_params


def main():
    constants.log_env_variables()
    params = load_params(constants.CONFIG_FILE_PATH)

    logger.info(
        "\n-------------------------------------- START PIPELINE --------------------------------------\n"
    )

    t0 = time.time()

    mission_factory = MissionFactory(params)
    mission = mission_factory.create_mission()
    # sampler = optuna.samplers.TPESampler()
    # study = optuna.create_study(direction="maximize", sampler=sampler, pruner=optuna.pruners.HyperbandPruner())
    # study.optimize(mission.execute, n_trials=15)
    mission.execute()

    # pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    # complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    #
    # print("Study statistics: ")
    # print("  Number of finished trials: ", len(study.trials))
    # print("  Number of pruned trials: ", len(pruned_trials))
    # print("  Number of complete trials: ", len(complete_trials))
    #
    # print("Best trial:")
    # trial = study.best_trial
    # print("  Value: ", trial.value)
    #
    # print("  Params: ")
    # for key, value in trial.params.items():
    #     print("   {}: {}".format(key, value))

    logger.info(
        "\n-------------------------------------- STOP PIPELINE --------------------------------------\n"
    )


if __name__ == "__main__":
    logger = setup_logger()
    main()
