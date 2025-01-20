import os
import time
import json
import uuid

import torch
from torch.utils.data import IterableDataset, Dataset
from torch import nn
from torch.nn import functional as f

from torchvision.transforms import ToPILImage

import numpy as np
from pathlib import Path
from rich.progress import Progress

import MalmoPython
import malmoutils
from lxml import etree

import networkx as nx
from skimage.morphology import dilation


torch.multiprocessing.set_sharing_strategy('file_system')

malmoutils.fix_print()

class EnvironmentGenerator(IterableDataset):
    def __init__(self, fn, port, batch_size=128, dataset_size=None, steps=50, tic_duration=0.008):
        super().__init__()
        self.tree = etree.parse(fn)
        self.batch_size = batch_size
        self.agent_host = MalmoPython.AgentHost()
        self.dataset_size = dataset_size
        self.current_samples = 0
        self.steps = steps
        self.tic_duration = tic_duration

        # Load environment
        self.env = MalmoPython.MissionSpec(etree.tostring(self.tree), True)

        # Do not record anything
        self.record = MalmoPython.MissionRecordSpec()

        # Initialize client pool
        pool = MalmoPython.ClientPool()
        info = MalmoPython.ClientInfo('localhost', port)
        pool.add(info)
        experiment_id = str(uuid.uuid1())

        # Initialize environment
        self.agent_host.startMission(
            self.env, pool, self.record, 0, experiment_id)

        # Loop until the mission starts
        world_state = self.agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()

        world_state = self.await_ws()
        if len(world_state.video_frames) == 0:
            time.sleep(0.1)
            world_state = self.await_ws()
        frame = world_state.video_frames[-1]
        self.HWC = (frame.height, frame.width, frame.channels)

        time.sleep(0.1)
        world_state = self.await_ws()

        self.start_time = time.time()

    def next_step(self):
        speed = 0.6 # I've changed this agent so that it has a greater speed to match what was indicated in the other code
        # visually it seems similar to my teleport agents
        ang_vel = np.random.normal(0.0, 2 * np.pi)

        velocity = {"speed": speed, "ang_vel": ang_vel}

        return velocity, False

    def __iter__(self):
        return self

    def __next__(self):
        H, W, C = self.HWC
        L = self.steps
        inputs = np.empty((L, H, W, C), dtype=np.uint8)
        actions = np.empty((L, 2), np.float32)
        state = np.empty((L, 3), dtype=np.float32)

        for idx in range(L):
            def btoa(pixels):
                return np.reshape(np.frombuffer(pixels, dtype=np.uint8), self.HWC)

            world_state = self.await_ws()
            pixels = world_state.video_frames[-1].pixels

            inputs[idx] = btoa(pixels).copy()
        
            msg = world_state.observations[-1].text
            x = float(json.loads(msg)["XPos"])
            z = float(json.loads(msg)["ZPos"])
            direction = int(json.loads(msg)["Yaw"])
            
            state[idx] = np.array([x, z, direction], dtype=np.float32)

            velocity, stuck = self.next_step()
            speed = velocity["speed"]
            ang_vel = velocity["ang_vel"]
            actions[idx] = [speed, ang_vel]

            for _ in range(6):
                self.agent_host.sendCommand(f"move {speed}")
                self.agent_host.sendCommand(f"turn {ang_vel}")

                time.sleep(self.tic_duration)

        return (inputs, actions, state)

    def generate_dataset(self, path: Path, size=1000):
        current_size = 0
        with Progress() as progress:
            task = progress.add_task("Building dataset...", total=size)
            while current_size < size:
                batch = self.__next__()
                if batch is None:
                    continue
                inputs, actions, state = batch
                current_path = path / f"{time.time()}"
                os.makedirs(current_path, exist_ok=True)
                for t in range(len(inputs)):
                    image = ToPILImage()(inputs[t])
                    image.save(current_path / f"{t}.png")
                np.savez(current_path / "actions.npz", actions)
                np.savez(current_path / "state.npz", state)

                current_size += self.steps
                progress.update(task, advance=self.steps)

    def await_ws(self, delay=0.001):
        world_state = self.agent_host.peekWorldState()
        while world_state.number_of_observations_since_last_state <= 0:
            time.sleep(delay)
            world_state = self.agent_host.peekWorldState()

        return world_state
