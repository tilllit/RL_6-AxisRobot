# 6-AxisRobot

<br />
This project uses Reinforcement Learning (RL) to teach a 6-Axis robot to drive on a given trajectory and reach a requested angle. <br />
The RL approach uses function approximation with a Neural Network (NN) to learn the trajectory. <br />
The used REINFORCE algorithm is based on an implementation from "Foundations of Deep Reinforcement learning"  by L Graesser and W Keng.

<br />

# Content

- [RL Task](#Task)
- [Environment](#Environment)
- [REINFORCE Algorithm](#Algorithm)
- [Neural Net](#NeuralNet)
- [Reward](#Reward)
- [Future Work](#FutureWork)


## Task

The specific task being solved by this project is to drive on a 10mm long trajectory through 3D space, while fulfilling a 20 degree rotation of the TCP (Tool Center Point) in reference to the Z-axis. <br />
<br />





Another requirement was to use RL (Reinforcement Learning): function approximation with a Neural Network as an approach to solve the problem. <br />




## Environment

The environment was designed to be compatible with the openai [Gymnasium-API](https://gymnasium.farama.org/api/env/). <br />
It has a reset and a step function and contains the agent, which is the robotic arm. <br />


### Robot Arm

The agent in the environment is a robotic arm with 6 joints. It's kinematic properties are defined in the following Denavit-Hartenberg matix: <br />

| joint angle [deg] | link length [m] | link offset [m] | link twist [deg] |
| :------- | :------: | :-------: | -------: |
| theta1 | 0 | 0.15185 | 90 |
| theta2 | -0.24355 | 0 | 0 |
| theta3 | -0.2132  | 0 | 0 |
| theta4 | 0 | 0.13105 | 90 |
| theta5 | 0 | 0.08535 | -90 |
| theta6 | 0 | 0.0921 | 0 |
<br />

The agent receives an array with angles for the 6 joints, performas a forward kinematic and returns the X, Y and Z position in 3D space as well as the 3 euler angles for the orientation of the TCP.
The resulting orientation can be plotted like this with the plot-function within the RobotArm-class:

<p align="center">
  <img height="400" width="450" src="Assets/RobotKinematic.png">
</p>


### Step

As action the step-function expects an array of 6 angle-deltas usualy with values of 0.1, 0 and -0.1 to increase or decrease the joint angles of the robot arm. <br />
It then updates the agents state and checks, if the TCP is still within a defined observation-space. <br />
If the position exceeds the boundarys of the observation space, a reset is triggered. <br />
Otherwise it calculates and returns the reward for the updated state. <br />
Example:

### Trajectory


## Algorithm

The used algorithm

## NeuralNet

## Reward

## FutureWork

<p align="center">
  <img height="640" width="960" src="Assets/UntilTCP.gif">
</p>

gamma1 = 0.99
gamma2 = 0.70
