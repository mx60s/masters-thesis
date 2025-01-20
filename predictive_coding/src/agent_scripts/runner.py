import os
import signal
from datetime import datetime
from pathlib import Path
from generator_pc import EnvironmentGenerator
import argparse

def handler(signum, frame):
    raise TimeoutError()

def main(args):
    signal.signal(signal.SIGALRM, handler)

    os.environ["MALMO_XSD_PATH"] = args.malmo_xsd_path
    environment = EnvironmentGenerator(args.xml_path, args.seed, rotations=args.rotations)

    dataset_dir = Path(args.dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.iterations):
        print(f"Iteration {i}")
        print(datetime.now().strftime("%I:%M %p"))
        try:
            signal.alarm(args.timeout)
            environment.generate_dataset(dataset_dir)
            signal.alarm(0)
        except TimeoutError:
            print(f"Timeout occurred at iteration {i}, continuing")
            del environment
            environment = EnvironmentGenerator(args.xml_path, args.seed, rotations=args.rotations)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EnvironmentGenerator to create datasets.")

    home_dir = str(Path.home())

    parser.add_argument(
        "--malmo-xsd-path", 
        type=str, 
        default=f"{home_dir}/malmo/Schemas", 
        help="Path to the MALMO XSD schemas."
    )
    parser.add_argument(
        "--xml-path", 
        type=str, 
        default=f"{home_dir}/malmo/small-world/natural.xml", 
        help="Path to the XML file defining the environment."
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=10001, 
        help="Seed for the EnvironmentGenerator."
    )
    parser.add_argument(
        "--rotations", 
        type=int, 
        default=0, 
        help="Number of rotations for the environment (default: 0)."
    )
    parser.add_argument(
        "--dataset-dir", 
        type=str, 
        default=f"{home_dir}/malmo/Python_Examples/data/small-world-pcs", 
        help="Directory to save the generated dataset."
    )
    parser.add_argument(
        "--iterations", 
        type=int, 
        default=1, 
        help="Number of iterations to run the dataset generation."
    )
    parser.add_argument(
        "--timeout", 
        type=int, 
        default=300, 
        help="Timeout in seconds for each iteration."
    )

    args = parser.parse_args()
    main(args)