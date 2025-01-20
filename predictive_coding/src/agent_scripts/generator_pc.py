import os
import time
import json
import uuid
import math
import sys
import shutil
import signal

import torch
from torch.utils.data import IterableDataset
from torchvision.transforms import ToPILImage

import numpy as np
from pathlib import Path
from rich.progress import Progress

import MalmoPython
import malmoutils
from lxml import etree

torch.multiprocessing.set_sharing_strategy("file_system")
malmoutils.fix_print()


def parse_boundaries(env_tree):
    """
    Parse the environment XML to grab the boundaries.
    Args:
    - env_tree: XML tree
    """
    ns = {'ns': 'http://ProjectMalmo.microsoft.com'}

    min_element = env_tree.find('.//ns:min', ns)
    max_element = env_tree.find('.//ns:max', ns)

    if min_element is not None and max_element is not None:
        min_coords = {
            'x': float(min_element.attrib['x']),
            'z': float(min_element.attrib['z'])
        }
        max_coords = {
            'x': float(max_element.attrib['x']),
            'z': float(max_element.attrib['z'])
        }
        return min_coords, max_coords
    else:
        raise ValueError("Could not find min or max elements in the XML.")


class EnvironmentGenerator(IterableDataset):
    def __init__(self, fn, port, tic_duration=0.008, rotations=17):
        """
        Set up a Malmo testing environment according to specs.
        This agent collects observations at each indicated direction in every 
        valid spot in the environment.
        Args:
        - fn: which mission to run
        - port: which port to contact the Minecraft server on
        - tic_duration: tic duration for the server (leave alone)
        - rotations: number of observations collected, evenly spaced along 360 degrees 
        """
        super().__init__()

        self.tree = etree.parse(fn)
        self.agent_host = MalmoPython.AgentHost()
        self.tic_duration = tic_duration
        self.rotations = rotations

        self.threshold = 0.001    # Changes in state above this are recorded
        self.render_tolerance = 0.05  # Allowed diff between render and recorded state
        self.visited = set()      # Locations visited so far

        # Load environment
        self.env = MalmoPython.MissionSpec(etree.tostring(self.tree), True)

        # Do not record anything
        self.record = MalmoPython.MissionRecordSpec()

        # Initialize client pool
        pool = MalmoPython.ClientPool()
        info = MalmoPython.ClientInfo("localhost", port)
        pool.add(info)
        experiment_id = str(uuid.uuid1())

        # Initialize environment
        self.agent_host.startMission(self.env, pool, self.record, 0, experiment_id)

        # Loop until the mission starts
        world_state = self.agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()

        time.sleep(5)
        world_state = self.wait_initial_state()
        frame = world_state.video_frames[-1]
        self.HWC = (frame.height, frame.width, frame.channels)

        self.locs = []

        min_bounds, max_bounds = parse_boundaries(self.tree)

        # Ensure integer bounds for range function
        min_x = int(math.floor(min_bounds['x']))
        max_x = int(math.ceil(max_bounds['x']))
        min_z = int(math.floor(min_bounds['z']))
        max_z = int(math.ceil(max_bounds['z']))

        self.locs.append(f'{min_x + 0.5}_{min_z + 0.5}')
        print(self.locs)

        #for x in range(min_x, max_x + 1):
        #    for z in range(min_z, max_z + 1):
        #        self.locs.append(f'{x + 0.5}_{z + 0.5}')

        self.start_time = time.time()

    def __iter__(self):
        return self

    def __next__(self):
        H, W, C = self.HWC
        inputs = np.empty((self.rotations, H, W, C), dtype=np.uint8)

        state = []
        actions = []

        if len(self.locs) > 0:
            loc = self.locs.pop(0).split('_')
            x = float(loc[0])
            z = float(loc[1])
        else:
            raise StopIteration

        base_x, base_z = x, z
        yaw = prev_yaw = 0

        self.curr_loc = (base_x, base_z, yaw)

        for idx in range(self.rotations):
            print(idx)
            def btoa(pixels):
                return np.reshape(np.frombuffer(pixels, dtype=np.uint8), self.HWC)

            self.agent_host.sendCommand(f"tp {x} 4.0 {z}")
            self.agent_host.sendCommand(f"setYaw {yaw}")
            time.sleep(1.7)  # Arbitrary, but this works best

            self.expected_x = x
            self.expected_y = 4.0  # Prevent placement in water or on top of a fence
            self.expected_z = z
            self.expected_yaw = yaw

            self.require_yaw_change = yaw != getattr(self, 'curr_yaw', yaw)
            self.require_move = x != getattr(self, 'curr_x', x) or z != getattr(self, 'curr_z', z)

            self.prev_state, error = self.wait_next_state()
            if error or not self.prev_state.video_frames:
                print(error)
                raise RuntimeError("Error in wait_next_state")

            pixels = self.prev_state.video_frames[-1].pixels

            inputs[idx] = btoa(pixels).copy()

            state.append((self.curr_x_from_render, self.curr_z_from_render, self.curr_yaw_from_render))

            step = np.random.uniform(-0.03, 0.03, size=2)  # Add a little noise to the position

            actions.append((step[0], step[1], prev_yaw - yaw))

            x, z = base_x + step[0], base_z + step[1]
            prev_yaw = yaw
            yaw = (idx + 1) * (360 / self.rotations)
            yaw %= 360

        state = np.array(state)
        actions = np.array(actions)

        return inputs, actions, state

    def generate_dataset(self, path: Path):
        # Can get stuck sometimes, set up a timeout handler
        def handler(signum, frame):
            raise TimeoutError()

        signal.signal(signal.SIGALRM, handler)

        # Clean up any partially-filled directories
        print("Checking for completed coordinates")
        orig_size = len(self.locs)
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                if f'{self.rotations - 1}.png' in os.listdir(item_path):
                    coord = item
                    if coord in self.locs:
                        self.locs.remove(coord)
                else:
                    print(f"Deleting the partial {item}")
                    shutil.rmtree(item_path)

        print(f"{len(self.locs)} / {orig_size} left to generate")

        print("Building dataset")

        with Progress() as progress:
            task = progress.add_task("Building dataset...", total=len(self.locs))

            while len(self.locs) > 0:
                try:
                    signal.alarm(100)
                    batch = self.__next__()
                    signal.alarm(0)
                except TimeoutError:
                    print(f"Timeout on {self.curr_loc}")
                    continue
                except StopIteration:
                    print("Finished")
                    return
                except Exception as e:
                    print(f"Exception: {e}")
                    continue

                inputs, actions, state = batch
                current_path = path / f"{state[0, 0]}_{state[0, 1]}"

                try:
                    os.makedirs(current_path, exist_ok=False)
                except FileExistsError:
                    continue

                for t in range(len(inputs)):
                    image = ToPILImage()(inputs[t])
                    image.save(current_path / f"{t}.png")

                np.savez(current_path / "actions.npz", actions)
                np.savez(current_path / "state.npz", state)

                progress.update(task, advance=1)

    def wait_initial_state(self):
        """Wait for the initial observation of the world and a frame."""
        # Wait for a valid observation
        world_state = self.agent_host.peekWorldState()
        while world_state.is_mission_running and all(
            e.text == "{}" for e in world_state.observations
        ):
            time.sleep(0.1)
            world_state = self.agent_host.peekWorldState()

        # Wait for a frame to arrive after that
        num_frames_seen = world_state.number_of_video_frames_since_last_state
        while (
            world_state.is_mission_running
            and world_state.number_of_video_frames_since_last_state == num_frames_seen
        ):
            time.sleep(0.1)
            world_state = self.agent_host.peekWorldState()

        world_state = self.agent_host.getWorldState()

        if world_state.is_mission_running:
            assert len(world_state.video_frames) > 0, "No video frames!"

            obs = json.loads(world_state.observations[-1].text)
            self.prev_x = obs["XPos"]
            self.prev_y = obs["YPos"]
            self.prev_z = obs["ZPos"]
            self.prev_yaw = obs["Yaw"]
            self.curr_x = self.prev_x
            self.curr_z = self.prev_z
            self.curr_yaw = self.prev_yaw
            self.prev_dir = self.prev_yaw  # The direction of movement and yaw are detangled

            self.prev_state = world_state

        return world_state

    def wait_next_state(self):
        """Wait for the observation to change as expected and a frame after each command."""
        obs = None
        wait = 0
        while wait < 10000:
            wait += 1
            world_state = self.agent_host.peekWorldState()
            if not world_state.is_mission_running:
                break
            if not all(e.text == "{}" for e in world_state.observations):
                obs = json.loads(world_state.observations[-1].text)
                self.curr_x = obs["XPos"]
                self.curr_y = obs["YPos"]
                self.curr_z = obs["ZPos"]
                self.curr_yaw = obs["Yaw"] % 360
                if self.require_move:
                    if (
                        abs(self.curr_x - self.prev_x) > self.threshold
                        or abs(self.curr_y - self.prev_y) > self.threshold
                        or abs(self.curr_z - self.prev_z) > self.threshold
                    ):
                        break
                elif self.require_yaw_change:
                    if abs(self.curr_yaw - self.prev_yaw) > self.threshold:
                        break
                else:
                    break
            time.sleep(self.tic_duration)

        if wait == 10000:
            print("Hung waiting for observation")
            return world_state, True

        # Wait for the render position to have changed
        wait = 0
        while wait < 70000:
            wait += 1
            world_state = self.agent_host.peekWorldState()
            if not world_state.is_mission_running:
                print("Mission ended.")
                break
            if len(world_state.video_frames) > 0:
                frame = world_state.video_frames[-1]
                self.curr_x_from_render = frame.xPos
                self.curr_y_from_render = frame.yPos
                self.curr_z_from_render = frame.zPos
                self.curr_yaw_from_render = frame.yaw % 360
                if self.require_move:
                    if (
                        abs(self.curr_x_from_render - self.prev_x) > self.threshold
                        or abs(self.curr_y_from_render - self.prev_y) > self.threshold
                        or abs(self.curr_z_from_render - self.prev_z) > self.threshold
                    ):
                        break
                elif self.require_yaw_change:
                    if abs(self.curr_yaw_from_render - self.prev_yaw) > self.threshold:
                        break
                else:
                    break
            time.sleep(self.tic_duration)

        if wait == 70000:
            print("Hung waiting for render")
            return world_state, True

        num_frames_before_get = len(world_state.video_frames)
        world_state = self.agent_host.getWorldState()

        if world_state.is_mission_running:
            assert len(world_state.video_frames) > 0, "No video frames!"
            num_frames_after_get = len(world_state.video_frames)
            assert (
                num_frames_after_get >= num_frames_before_get
            ), "Fewer frames after getWorldState!?"
            frame = world_state.video_frames[-1]
            if obs is None and len(world_state.observations) > 0:
                obs = json.loads(world_state.observations[-1].text)
                self.curr_x = obs["XPos"]
                self.curr_y = obs["YPos"]
                self.curr_z = obs["ZPos"]
                self.curr_yaw = obs["Yaw"] % 360

            if (
                abs(self.curr_x - self.expected_x) > self.render_tolerance
                or abs(self.curr_y - self.expected_y) > self.render_tolerance
                or abs(self.curr_z - self.expected_z) > self.render_tolerance
                or abs(self.curr_yaw - self.expected_yaw) > self.render_tolerance
            ):
                print(
                    " - ERROR DETECTED! Expected:",
                    self.expected_x,
                    ",",
                    self.expected_y,
                    ",",
                    self.expected_z,
                    "yaw",
                    self.expected_yaw,
                )
                print(self.curr_x, self.curr_y, self.curr_z, self.curr_yaw)
                return world_state, True

            self.curr_x_from_render = frame.xPos
            self.curr_y_from_render = frame.yPos
            self.curr_z_from_render = frame.zPos
            self.curr_yaw_from_render = frame.yaw % 360
            if (
                abs(self.curr_x_from_render - self.expected_x) > self.render_tolerance
                or abs(self.curr_y_from_render - self.expected_y) > self.render_tolerance
                or abs(self.curr_z_from_render - self.expected_z) > self.render_tolerance
                or abs(self.curr_yaw_from_render - self.expected_yaw) > self.render_tolerance
            ):
                print(
                    " - ERROR DETECTED! Expected:",
                    self.expected_x,
                    ",",
                    self.expected_y,
                    ",",
                    self.expected_z,
                    "yaw",
                    self.expected_yaw,
                )
                print(
                    self.curr_x_from_render,
                    self.curr_y_from_render,
                    self.curr_z_from_render,
                    self.curr_yaw_from_render,
                )
                return world_state, True

            self.prev_x = self.curr_x
            self.prev_y = self.curr_y
            self.prev_z = self.curr_z
            self.prev_yaw = self.curr_yaw

        return world_state, False

