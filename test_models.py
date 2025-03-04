import logging
from logging import getLogger, DEBUG
import yaml

from models import get_model

logging.basicConfig(level=DEBUG)
log = getLogger()


def main():
    config_file = "./configs/dscgrapher.yaml"

    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model = get_model(config["model"], config, log)
    log.info(str(model))


if __name__ == "__main__":
    main()
