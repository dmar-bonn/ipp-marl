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

    mission_factory = MissionFactory(params)
    mission = mission_factory.create_mission()
    mission.execute()

    logger.info(
        "\n-------------------------------------- STOP PIPELINE --------------------------------------\n"
    )


if __name__ == "__main__":
    logger = setup_logger()
    main()
