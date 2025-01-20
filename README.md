#From Sensory Input to Cognitive Maps: Exploring the Significance of Spatial Representations in Artificial Hippocampal Models

## Table of contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Methods](#methods)
5. [Results](#results)
6. [Acknowledgments](#acknowledgments)

## Overview

This repo contains work that I did for my Master's thesis in Computer Science at UT Austin, under the supervision of Dr. Xue-Xin Wei and Dr. Risto Miikkulainen. 

You can read the thesis here: [link](mvonebers.com/docs/thesis.pdf)
I also wrote a [small article for Nature Machine Intelligence](https://www.nature.com/articles/s42256-024-00885-9) with my advisor about the work by Gornet and Thomson that this thesis is based on.

This project investigates and critiques two recent models of hippocampal place cells and the cognitive maps that they are hypothesized to support. 

Several of the notebooks include notes at the bottom with reflections on further work I'd like to do. There's still a lot of work to be done here in terms of cleaning up!

## Installation

Instructions regarding the setup for the [Minecraft Malmo](https://github.com/microsoft/malmo) package will be included in the future.

Please run setup.py to install packages needed for predictive_coding and dnn_place_cells.

## Usage

Right now only Steering.ipynb works from scratch, sorry! Working on the rest of the demonstration notebooks now.

## Methods and Results

A short summary of most of the work I did for this project:
- I originally implemented the entire predictive coding project from scratch, then switched over to using Gornet & Thomson's solution
- I implemented a few versions of a modified predictive coder, called VestibularCoder, which, in addition to taking a sequence of image observations as input, takes information about the movement of the agent for each step (velocity and heading direction). This model achieved better performance on the next-step prediction task, but wasn't able to use this information in an integrated map of space. For this demonstration, see notebooks/Steering.ipynb
- The original location probes were linear and were trained on the entire dataset. They also were only trained on one viewing angle. I implemented and trained a nonlinear probe across multiple viewing angles, as well as a Linear SVM, in the same vein as Gulli et al. 2020. Performance for those was changed to report on a held-out validation set.
- I made a bunch of Minecraft agents which can move in different ways. This includes but isn't limited to a swivelling head (in order to study the effects of detangling head direction and movement direction) and obstacle avoidance. To detangle heading I had to exploit the teleportation feature instead of just using the movement guide for Malmo.
- I tried isolating features with a sparse autoencoder to see if I could find anything relating to a cognitive map, but just came up with a large assortment of high-frequency visual features. I'll add this notebook back in the future just for fun.
- I updated the place cell discrimination code in the Luo et al. project to include the full method from Tanni et al. This includes making sure the place fields are smooth and stable features. I also fixed a bug in the Luo et al. codebase which was indicating mixed encoding where there wasn't any.
- I extended the Luo et al. code to be able to classify and lesion units from the Predictive Coding model
- I rebuilt the Luo et al. Unity environment in Minecraft and adjusted the code to take the new source so that all models could be tested in the same regime.

For now, please feel free to peruse results in the thesis doc [here.](mvonebers.com/docs/thesis.pdf)

## Acknowledgements

This repository includes contributions from:
- [Gornet and Thomson (2024)](https://github.com/jgornet/predictive-coding-recovers-maps) (MIT License)
- [Luo et al. (2024)](https://github.com/don-tpanic/Space). (MIT License)

All modifications and new code are licensed under the MIT License unless otherwise noted. Any file which contains code from one of these two sources is indicated with a comment at the top.
See `/predictive-coding/LICENSE` and `/dnn-place-cells/LICENSE` for details. 
Thank you so much to the authors for your lovely open sourced projects!