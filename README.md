# 6-AxisRobot

<br />
This project uses Reinforcement Learning (RL) to teach a 6-Axis robot to drive on a given trajectory and reach a requested angle. <br />
The RL approach uses function approximation with a Neural Network (NN) to learn the trajectory. <br />
The used REINFORCE algorithm is based on an implementation from "Foundations of Deep Reinforcement learning"  by L Graesser and W Keng.

<br />

# Content

- [Environment](#Environment)
- [REINFORCE Algorithm](#Algorithm)
- [Neural Net](#NeuralNet)
- [Reward](#Reward)
- [Future Work](#FutureWork)


## Environment

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

The agent receives an array with angles for the 6 joints and returns the x, y and z position in 3D space as well as the 3 euler angles for the orientation of the TCP.
The resulting orientation can be plotted like this:

<p align="center">
  <img height="400" width="450" src="Assets/RobotKinematic.png">
</p>

## Algorithm

## NeuralNet

## Reward

## FutureWork

<p align="center">
  <img height="640" width="960" src="Assets/UntilTCP.gif">
</p>

gamma1 = 0.99
gamma2 = 0.70
