import argparse
import multiprocessing

from src.config_manager import ConfigManager


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--confdir', type=str, default='./res/config.yml', help='Path to the config file')
    return parser.parse_args()


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    args = get_args()
    config_parser = ConfigManager(args.confdir)
    config_parser.run()
