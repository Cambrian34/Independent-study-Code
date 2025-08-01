# Autonomous Robot: LiDAR Navigation & Vision-Based Object Recognition

## Overview

This repository contains code and resources for an independent research project focused on developing an autonomous robot. The robot leverages LiDAR for navigation and computer vision for real-time object recognition.

##Hardware Components
- **LiDAR Sensor:** For mapping and obstacle detection.
- **Camera:** For visual perception and object recognition.
- **Microcontroller:** For processing sensor data and controlling the robot's movement.
- **Power Supply:** Battery or power source to run the robot.
- **Chassis:** Physical structure to house components and provide mobility.
- **Wheels/Tracks:** For movement across various terrains.


## Features

- **LiDAR Navigation:** Real-time mapping and obstacle avoidance using LiDAR sensors.
- **Vision-Based Object Recognition:** Object detection and classification with deep learning models.
- **Modular Codebase:** Separate modules for sensor data processing, navigation, and control.


## Project Structure

- `src/` — Source code for navigation, perception, and control modules.
- `models/` — Pre-trained and custom-trained models for object recognition.
- `docs/` — Documentation, research notes, and setup guides.
- `requirements.txt` — Python dependencies for development and deployment.

## Getting Started

1. Clone this repository.
2. Install dependencies:  
    `pip install -r requirements.txt`
3. Set up your environment and hardware as described in `docs/setup.md`.
4. Connect LiDAR and camera hardware.
5. Run the main script to start the robot stack.

## Research Objectives

- Fuse LiDAR and movemnt controller for robust autonomous navigation.
- Object recognition accuracy in real-world and simulated environments.
- Explore improvements in sensor integration and real-time performance.

## Acknowledgements

This independent study was supported by mentor acknowledged in `docs/acknowledgements.md`.

## License

For educational and research use only. See `LICENSE` for details.
