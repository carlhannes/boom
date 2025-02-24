# AI Coding Agent

An AI coding agent based on Learn-by-interact principles that autonomously performs coding tasks in Git repositories.

## Project Overview

This project implements an autonomous coding agent that can perform tasks in Git repositories using a Learn-by-interact approach. The agent learns from its interactions with the codebase and improves its performance through trajectory-based learning.

## Core Components

### 1. Coding Agent (`ai_agent/core/agent.py`)
The main agent class that orchestrates task execution, environment interaction, and learning processes.

### 2. Git Environment (`ai_agent/environment/git_env.py`)
Provides an interface to interact with Git repositories, handling file operations, status checks, and Git commands.

### 3. Self Learner (`ai_agent/core/learner.py`)
Implements the learning mechanisms, including task generation from documentation and backward construction of instructions.

### 4. Trajectory Management (`ai_agent/data/trajectory_manager.py`)
Handles storage and retrieval of execution trajectories, enabling learning from past experiences.

## Key Features

- Task generation from documentation
- Execution trajectory recording
- Backward construction of instructions
- Similarity-based trajectory retrieval
- Git repository interaction
- OpenAI GPT integration

## Installation

The project uses pip for dependency management. Key dependencies are specified in `requirements.txt`.

Required Python version: ^3.9

Main dependencies:
- openai ^1.0.0
- gitpython ^3.1.0
- numpy ^2.2.3
- scipy ^1.13.1
- pytest ^8.3.4

## Getting Started

See the `examples/demo.py` file for example usage of the agent.