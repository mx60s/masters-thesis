import os
import time
import json
import uuid
import random
import math
import sys

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


torch.multiprocessing.set_sharing_strategy("file_system")

malmoutils.fix_print()


class EnvironmentGenerator(IterableDataset):
    def __init__(
        self, fn, port, batch_size=128, dataset_size=None, steps=50, tic_duration=0.008, momentum_coeff=0.1, dir_inertia_coeff=0.1
    ):
        super().__init__()
        self.tree = etree.parse(fn)
        self.batch_size = batch_size
        self.agent_host = MalmoPython.AgentHost()
        self.dataset_size = dataset_size
        self.current_samples = 0
        self.steps = steps
        self.tic_duration = tic_duration
        self.tolerance = 0.001
        self.render_tolerance = 0.05
        self.border_region = 0.5 #25
        self.momentum_coeff = momentum_coeff
        self.dir_inertia_coeff = dir_inertia_coeff

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

        world_state = self.wait_initial_state()
        frame = world_state.video_frames[-1]
        self.HWC = (frame.height, frame.width, frame.channels)

        self.start_time = time.time()

    def init_pathfinding(self):
        grid = [
            block == "air"
            for block in json.loads(self.prev_state.observations[-1].text)["board"]
        ]
        grid = ~np.array(grid).reshape((66, 41))
        grid = np.flip(grid, axis=1)
        
        self.grid = grid
        
        # Collect all of the positions of obstacles in the env
        obstacle_locs = []
        
        for i in range(66):
            for j in range(41):
                if grid[i, j]:
                    obstacle_locs.append((-j + 19.5, i - 29.5))

        for z in [-30, 35]:
            for x in range(-20, 20):
                obstacle_locs.append((x, z))
        
        for x in [-20, 20]:
            for z in range(-30, 35):
                obstacle_locs.append((x, z))
                
        self.obstacle_locs = set(obstacle_locs)

    
    def get_nearest_obstacle(self, position):
        '''
        Find closest obstacle to the current position
        '''
        
        min_dist = float('inf')
        min_obst = None
        for obst in self.obstacle_locs:
            obst = np.array(obst, dtype=np.float32)
            d = np.linalg.norm(position-obst)
            if d < min_dist:
                min_dist = d
                min_obst = obst
          
        return min_obst, min_dist

    
    def avoid_obstacles(self, position, move_dir):
        '''
        Compute distance and angle to nearest obstacle/boundary
        '''
        x, z = position
        
        obst, d_obst = self.get_nearest_obstacle(position)
        
        def get_angle(x, z):
            return np.arctan2(-x, z)
        
        a_obst = get_angle(obst[0] - x, obst[1] - z)
        a_obst = np.mod(a_obst - move_dir + np.pi, 2*np.pi) - np.pi

        is_near_obst = (d_obst < self.border_region)*(np.abs(a_obst) < np.pi/2)
        turn_angle = np.zeros_like(move_dir)
        turn_angle[is_near_obst] = -np.sign(a_obst) * np.pi / 4 * (1 - d_obst / self.border_region)
        return is_near_obst, turn_angle

    def generate_path(self, momentum_coeff=None, dir_inertia_coeff=None):
        
        if not momentum_coeff:
            momentum_coeff = self.momentum_coeff
        if not dir_inertia_coeff:
            dir_inertia_coeff = self.dir_inertia_coeff
        
        dt = 0.3  # time step increment (seconds)
        sigma = 1.76 * 3 * momentum_coeff  # stdev rotation velocity (rads/sec)
        b = 0.23 * 1 * np.pi * dir_inertia_coeff # forward velocity rayleigh dist scale (m/sec)
        
        num_samples = self.steps
        
        # Find initial positions
        position = np.zeros([num_samples+2, 2]).astype(np.float32)
        move_dir = np.zeros([num_samples+2]).astype(np.float32)
        yaw = np.zeros([num_samples+2]).astype(np.float32)

        locs = ["112.5_197.5", "112.5_211.5", "112.5_214.5", "113.5_212.5", "113.5_221.5", "114.5_200.5", "115.5_217.5", "116.5_208.5", "117.5_215.5", "117.5_216.5", "118.5_214.5", "118.5_220.5", "119.5_201.5", "119.5_220.5", "120.5_210.5", "120.5_218.5", "122.5_199.5", "123.5_210.5", "126.5_203.5", "126.5_206.5", "126.5_209.5", "126.5_219.5", "127.5_206.5", "127.5_210.5", "129.5_219.5", "130.5_198.5", "132.5_202.5", "132.5_216.5", "133.5_202.5", "134.5_221.5", "135.5_201.5"]
        
        idx = np.random.randint(0, len(locs))
        pos = locs[idx].split('_')
        
        position[0, 0] = float(pos[0])
        position[0, 1] = float(pos[1])
        move_dir[0] = np.random.uniform(0, 2*np.pi, 1).astype(np.float32)
        yaw[0] = 0
        velocity = np.zeros([num_samples+2]).astype(np.float32)
        ang_v = np.zeros([num_samples+2]).astype(np.float32)

        # Generate sequence of random boosts and turns
        random_turn = np.random.normal(0, sigma, [num_samples+1]).astype(np.float32)
        random_vel = np.random.normal(0, b, [num_samples+1]).astype(np.float32)
        random_vel = np.abs(random_vel)
        v = np.abs(np.random.normal(0, b*np.pi/2, 1)).astype(np.float32)

        for t in range(num_samples + 1):
            v = 0.7 + random_vel[t]
            turn_angle = 0
            
            is_near_obst, turn_angle = self.avoid_obstacles(position[t], move_dir[t])
            
            if is_near_obst:
                v *= 0.45 
            
            turn_angle += dt*random_turn[t]
            
            # Take a step
            velocity[t] = v*dt
            update = velocity[t]*np.stack([-np.sin(move_dir[t]), np.cos(move_dir[t])], axis=-1)
            position[t+1] = position[t] + update
                    
            move_dir[t+1] = move_dir[t] + turn_angle
            
            yaw[t+1] = math.fmod(move_dir[t] / np.pi * 180, 360) # move yaw to end
        
        move_dir = np.mod(move_dir + np.pi, 2*np.pi) - np.pi
        ang_v = np.diff(move_dir, axis=-1)
        
        return velocity[:num_samples], ang_v[:num_samples], position[:num_samples], move_dir[:num_samples], yaw[:num_samples]

    def __iter__(self):
        return self

    def __next__(self):
        H, W, C = self.HWC
        L = self.steps
        inputs = np.empty((L, H, W, C), dtype=np.uint8)
 
        speed, ang_vel, position, move_dir, yaw = self.generate_path()
    
        move_dir = np.expand_dims(move_dir, axis=-1)
        actions = np.column_stack((speed, ang_vel))
        state = []
        
        for idx in range(L):
            def btoa(pixels):
                return np.reshape(np.frombuffer(pixels, dtype=np.uint8), self.HWC)

            new_x, new_z = position[idx]
            new_yaw = yaw[idx]
            
            self.agent_host.sendCommand(f"tp {new_x} 4.0 {new_z}")
            self.agent_host.sendCommand(f"setYaw 0")
            
            time.sleep(0.06)

            self.expected_x = new_x
            self.expected_y = 4.0
            self.expected_z = new_z
            self.expected_yaw = 0
            self.require_yaw_change = False
            self.require_move = new_x != self.curr_x or new_z != self.curr_z
            
            self.prev_state, error = self.wait_next_state()
            if error:
                return None
            
            # Fill batch
            if not self.prev_state.video_frames:
                return None
            pixels = self.prev_state.video_frames[-1].pixels
            
            state.append((self.curr_x_from_render, self.curr_z_from_render, self.curr_yaw_from_render))

            inputs[idx] = btoa(pixels).copy()
        
        state = np.stack(state)
        
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

    # turn waiting from malmo python examples
    def wait_initial_state(self):
        """Before a command has been sent we wait for an observation of the world and a frame."""
        # wait for a valid observation
        world_state = self.agent_host.peekWorldState()
        while world_state.is_mission_running and all(
            e.text == "{}" for e in world_state.observations
        ):
            world_state = self.agent_host.peekWorldState()
        # wait for a frame to arrive after that
        num_frames_seen = world_state.number_of_video_frames_since_last_state
        while (
            world_state.is_mission_running
            and world_state.number_of_video_frames_since_last_state == num_frames_seen
        ):
            world_state = self.agent_host.peekWorldState()
        world_state = self.agent_host.getWorldState()

        if world_state.is_mission_running:

            assert len(world_state.video_frames) > 0, "No video frames!?"

            obs = json.loads(world_state.observations[-1].text)
            self.prev_x = obs["XPos"]
            self.prev_y = obs["YPos"]
            self.prev_z = obs["ZPos"]
            self.prev_yaw = obs["Yaw"]
            self.curr_x = self.prev_x
            self.curr_z = self.prev_z
            self.curr_yaw = self.prev_yaw
            self.prev_dir = (
                self.prev_yaw
            )  # the direction of movement and yaw are detangled
            #print(
            #    "Initial position:",
            #    self.prev_x,
            #    ",",
            #    self.prev_y,
            #    ",",
            #    self.prev_z,
            #    "yaw",
            #    self.prev_yaw,
            #)

            self.prev_state = world_state
            self.init_pathfinding()

        return world_state

    def wait_next_state(self):
        """After each command has been sent we wait for the observation to change as expected and a frame."""
        # wait for the observation position to have changed
        #print("Waiting for observation...", end=" ")
        obs = None
        wait = 0
        while wait < 10000:
            wait += 1
            world_state = self.agent_host.peekWorldState()
            if not world_state.is_mission_running:
                #print("mission ended.")
                break
            if not all(e.text == "{}" for e in world_state.observations):
                obs = json.loads(world_state.observations[-1].text)
                self.curr_x = obs["XPos"]
                self.curr_y = obs["YPos"]
                self.curr_z = obs["ZPos"]
                self.curr_yaw = math.fmod(obs["Yaw"], 360)
                if self.require_move:
                    if (
                        math.fabs(self.curr_x - self.prev_x) > self.tolerance
                        or math.fabs(self.curr_y - self.prev_y) > self.tolerance
                        or math.fabs(self.curr_z - self.prev_z) > self.tolerance
                    ):
                        break
                elif self.require_yaw_change:
                    if math.fabs(self.curr_yaw - self.prev_yaw) > self.tolerance:
                        break
                else:
                    break

        if wait == 10000:
            print("hung waiting for obs")
            return world_state, True
        # wait for the render position to have changed
        #print("Waiting for render...", end=" ")
        
        wait = 0
        while wait < 70000:
            wait += 1
            world_state = self.agent_host.peekWorldState()
            if not world_state.is_mission_running:
                #print("mission ended.")
                break
            if len(world_state.video_frames) > 0:
                # #print('render changed')
                frame = world_state.video_frames[-1]
                self.curr_x_from_render = frame.xPos
                self.curr_y_from_render = frame.yPos
                self.curr_z_from_render = frame.zPos
                self.curr_yaw_from_render = math.fmod(frame.yaw, 360)
                if self.require_move:
                    # #print('render move required')
                    if (
                        math.fabs(self.curr_x_from_render - self.prev_x) > self.tolerance
                        or math.fabs(self.curr_y_from_render - self.prev_y) > self.tolerance
                        or math.fabs(self.curr_z_from_render - self.prev_z) > self.tolerance
                    ):
                        #   #print('render received a move.')

                        break
                elif self.require_yaw_change:
                    if math.fabs(self.curr_yaw_from_render - self.prev_yaw) > self.tolerance:
                        #  #print('render received a turn.')
                        break
                else:
                    # #print('render received.')
                    break

        if wait == 70000:
            print("hung waiting for render")
            return world_state, True            
                    
        num_frames_before_get = len(world_state.video_frames)
        world_state = self.agent_host.getWorldState()

        if world_state.is_mission_running:
            assert len(world_state.video_frames) > 0, "No video frames!?"
            num_frames_after_get = len(world_state.video_frames)
            assert (
                num_frames_after_get >= num_frames_before_get
            ), "Fewer frames after getWorldState!?"
            frame = world_state.video_frames[-1]
            # obs = json.loads( world_state.observations[-1].text )
            self.curr_x = obs["XPos"]
            self.curr_y = obs["YPos"]
            self.curr_z = obs["ZPos"]
            self.curr_yaw = math.fmod(
                obs["Yaw"], 360
            )  
                
            # check out of bounds
            #if self.curr_x <= -20 or self.curr_x >= 20 or self.curr_z <= -30 or self.curr_z >= 35:
            #    print("Out of bounds")
            #    return world_state, True
            
            if (
                math.fabs(self.curr_x - self.expected_x) > self.render_tolerance
                or math.fabs(self.curr_y - self.expected_y) > self.render_tolerance
                or math.fabs(self.curr_z - self.expected_z) > self.render_tolerance
                or math.fabs(self.curr_yaw - self.expected_yaw) > self.render_tolerance
            ):
             #   print(
             #       " - ERROR DETECTED! Expected:",
             #       self.expected_x,
             #       ",",
             #       self.expected_y,
             #       ",",
             #       self.expected_z,
             #       "yaw",
             #       self.expected_yaw,
             #   )
             #   print(self.curr_x, self.curr_y, self.curr_z, self.curr_yaw)
                 #sys.exit("expected vs curr issue")
                return world_state, True
            else:
                pass
            #   #print('as expected.')
            self.curr_x_from_render = frame.xPos
            self.curr_y_from_render = frame.yPos
            self.curr_z_from_render = frame.zPos
            # #print('rendered yaw', frame.yaw)
            self.curr_yaw_from_render = math.fmod(
                frame.yaw, 360
            )  
            if (
                math.fabs(self.curr_x_from_render - self.expected_x) > self.render_tolerance
                or math.fabs(self.curr_y_from_render - self.expected_y)
                > self.render_tolerance
                or math.fabs(self.curr_z_from_render - self.expected_z)
                > self.render_tolerance
                or math.fabs(self.curr_yaw_from_render - self.expected_yaw)
                > self.render_tolerance
            ):
              #  print(
              #      " - ERROR DETECTED! Expected:",
              #      self.expected_x,
              #      ",",
              #      self.expected_y,
              #      ",",
              #      self.expected_z,
              #      "yaw",
              #      self.expected_yaw,
              #  )
              #  print(self.curr_x, self.curr_y, self.curr_z, self.curr_yaw)
                
                # sys.exit("curr vs render issue")
                return world_state, True
            else:
                pass
            #   #print('as expected.')
            self.prev_x = self.curr_x
            self.prev_y = self.curr_y
            self.prev_z = self.curr_z
            self.prev_yaw = self.curr_yaw

        return world_state, False
