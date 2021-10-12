"""Arguments."""
import argparse


def parse_args():
    """Parse input arguments."""
    # Load config files
    parser = argparse.ArgumentParser(prog="MCPredictor")
    # Set basic arguments
    parser.add_argument("--data_dir", default="/home/jinxiaolong/bl/data/gandc16",
                        type=str, help="MCNC corpus directory")
    parser.add_argument("--work_dir", default="/home/jinxiaolong/bl/data/sent_event_data",
                        type=str, help="Workspace directory")
    parser.add_argument("--device", default="cuda:0",
                        choices=["cpu", "cuda:0", "cuda:1", "cuda:2", "cuda:3"],
                        help="Device used for models.")
    parser.add_argument("--mode", default="preprocess",
                        choices=["preprocess", "train", "dev", "test"],
                        type=str, help="Experiment mode")
    parser.add_argument("--model_config", default="config/scpredictor-sent.json",
                        type=str, help="Model configuration files")
    parser.add_argument("--multi", action="store_true", default=False)
    # Set model arguments
    return parser.parse_args()


CONFIG = parse_args()
