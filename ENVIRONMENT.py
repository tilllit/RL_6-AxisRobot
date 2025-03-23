import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Für ältere Matplotlib-Versionen
import matplotlib.gridspec as gridspec
from RobotKin import RobotArm

import datetime


class Env():
    def __init__(self, dh_params, init_angles):
        self.initAngles = np.array(init_angles, dtype=np.float32)
        self.dh_matrix = dh_params
        self.robot = RobotArm(dh_params, init_angles)
        self.Terminated = False
        self.Truncated = False
        self.info = {}
        self.Angles = self.initAngles.copy()
        self.cntStep = 0
        self.rewardHistory = []
        self.plotTraj = []
        self.mseHistory = []
        self.reachedAngle = False
        self.pltCnt = 0


    def reset(self):
        self.Angles = self.initAngles.copy()
        self.Terminated = False
        self.Truncated = False

        self.ReachedTCP = False
        self.rewardHistory = []
        #self.plotTraj = []
        self.reachedAngle = False
        self.mseHistory = []
        self.cntStep = 0
        return self.Angles

    def step(self, action_deltas, plot=False):
        """
        :param action_deltas: Vektor der Deltas für jedes Gelenk (direkt von der Policy)
        """
        self.cntStep += 1
        self.Angles = self.Angles + action_deltas
        message = str(action_deltas)
        Reward, MSE = self.reward()
        #Reward = self.rewardGPT()
        self.rewardHistory.append(Reward)
        self.mseHistory.append(MSE)
        # Überprüfen, ob der Roboter den erlaubten Raum verlässt


        if self.checkBounds():
            self.Terminated = True
            message += " ( RESET! )"
            
            # Periodischer plot
            if plot:
                fileName = f"Trajectory"
                startPos = np.array(self.robot.get_tcp_pose(self.initAngles)[:3])  # x,y,z der Startposition
                cubeSize = 0.01 / np.sqrt(3)
                Ospace = np.array([cubeSize, cubeSize, cubeSize])
                finish = startPos + Ospace
                optTraj = np.array([startPos, finish])
                self.render_trajectory(self.plotTraj ,optTraj, fileName)
        
        state = self.Angles
        
        return state, Reward, self.Terminated, self.Truncated, self.mseHistory


    def reward(self):
        reward = 0

        # Translation
        startPos = np.array(self.robot.get_tcp_pose(self.initAngles)[:3])  # x,y,z der Startposition
        aktPos   = np.array(self.robot.get_tcp_pose(self.Angles)[:3])
        cubeSize = 0.01 / np.sqrt(3)
        Ospace = np.array([cubeSize, cubeSize, cubeSize])
        finish = startPos + Ospace
        TRANS = False

        # Rotation
        startANG = np.array(self.robot.get_tcp_pose(self.initAngles)[4])
        aktANG   = np.array(self.robot.get_tcp_pose(self.Angles)[4])
        finishANG = startANG - 20   # Deg
        ROT = False
        
        self.plotTraj.append(self.robot.get_tcp_pose(self.Angles))
        optTraj = np.array([startPos, finish])
        

        # Translation Finish
        tolerance = 0.0005
        if np.linalg.norm(aktPos - finish) < tolerance:
            print("TCP reached Endpoint!")
            TRANS = True
            tmpReward = reward + 2000
            tmpReward = tmpReward - self.cntStep * 30
            
            if tmpReward < 300:
                reward = 300
            else:
                reward = tmpReward
 
            self.Terminated = True
            self.ReachedTCP = True
            now = datetime.datetime.now()
            fileName = f"TCP"
            self.render_trajectory(self.plotTraj, optTraj, fileName)

        # Rotation Finish
        tolerance = 1.5
        if (abs(finishANG - aktANG) < tolerance):
            ROT = True
            if not self.reachedAngle:
                print("Angle reached Endpoint!")
                reward += 300
                fileName = f"Angle"
                #self.render_trajectory(self.plotTraj, optTraj, fileName)
            else:
                reward += 100

            self.reachedAngle = True


        if TRANS and ROT:
            reward += 10000
            print("\nSuccess!!! Reached Endpoint & Angle")
            fileName = f"Success"
            self.render_trajectory(self.plotTraj, optTraj, fileName)


        # Translation reward
        v = finish - startPos
        w = aktPos - startPos
        t = np.dot(w, v) / np.dot(v, v)
        t_clamped = np.clip(t, 0, 1)
        reward = reward + t_clamped * 100

        projection = startPos + t_clamped * v
        distance = np.linalg.norm(aktPos - projection)
        reward = reward - distance * 100

        # Rotation reward
        if (reward > 0):
            difANG = 20 - abs(finishANG - aktANG)
            if difANG >0:
                reward += difANG * 100

        MSE = self.computeMSE(aktPos,aktANG , startPos, finish, finishANG)
        return reward, MSE
    

    def checkBounds(self):
        startPos = np.array(self.robot.get_tcp_pose(self.initAngles)[:3])
        aktPos   = np.array(self.robot.get_tcp_pose(self.Angles)[:3])
        margin = np.array([0.001, 0.001, 0.001])
        cubeSize = 0.01 / np.sqrt(3)
        Ospace = np.array([cubeSize, cubeSize, cubeSize])
        lower_bound = startPos - margin
        upper_bound = startPos + Ospace
        if np.any(aktPos < lower_bound) or np.any(aktPos > upper_bound):
            #print("!!!! Error - out of space !!!!")
            return True
        return False

    def computeMSE(self, currentPos, currentAngle, startPos, finish, finishAngle, weight_angle=1.0):
        v = finish - startPos
        t = np.dot(currentPos - startPos, v) / np.dot(v, v)
        P_closest = startPos + t * v
        error_pos = currentPos - P_closest
        mse_pos = np.mean(error_pos ** 2)
        
        error_angle = currentAngle - finishAngle
        mse_angle = error_angle ** 2
        mse_total = (mse_pos + weight_angle * mse_angle) / 2.0  # Division durch 2 für Durchschnitt
        return mse_total

    def render_trajectory(self, trajectory, optimTraj, filename="trajectory.png"):
        try:
            # Trajektorie in ein NumPy-Array umwandeln
            trajectory = np.array(trajectory)
            positions = trajectory[:, :3]  # x, y, z
            angles    = trajectory[:, 3:6]   # alpha, beta, gamma

            optimTraj = np.array(optimTraj)

            # Erstelle Figure mit GridSpec (2 Zeilen, 2 Spalten)
            # Höhe der oberen Zeile größer (z.B. Faktor 2), damit genug Platz für 3D-Plots ist
            fig = plt.figure(figsize=(16, 10))
            gs = gridspec.GridSpec(3, 2, height_ratios=[2, 1, 1])

            # 1) Oberer, linker Subplot: 3D-Ansicht
            ax1 = fig.add_subplot(gs[0, 0], projection='3d')
            ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                    color='red', marker='o', linestyle='-')
            ax1.plot(optimTraj[:, 0], optimTraj[:, 1], optimTraj[:, 2],
                    color='green', marker='o', linestyle='-')
            
            axis_length = 0.0005  # Länge der Achsen
            for i in range(len(positions)):
                x, y, z = positions[i]
                alpha, beta, gamma = angles[i]

                # Rotationsmatrix aus Euler-Winkeln
                R = self.robot.euler_to_rotation_matrix(alpha, beta, gamma)

                # Lokale Einheitsvektoren für x, y, z (im Endeffektor-KS)
                ex_local = np.array([1, 0, 0])
                ey_local = np.array([0, 1, 0])
                ez_local = np.array([0, 0, 1])

                # In den Welt-Raum transformieren und auf axis_length skalieren
                ex_global = R @ ex_local * axis_length
                ey_global = R @ ey_local * axis_length
                ez_global = R @ ez_local * axis_length

                # # Ursprung jeder kleinen Achse ist (x, y, z).
                # # Zeichne drei Linien: (x->x+ex_global), (y->y+ey_global) usw.
                # ax1.plot([x, x + ex_global[0]],
                #         [y, y + ex_global[1]],
                #         [z, z + ex_global[2]],
                #         color='blue', linewidth=2)

                ax1.plot([x, x + ey_global[0]],
                        [y, y + ey_global[1]],
                        [z, z + ey_global[2]],
                        color='orange', linewidth=1)

                # ax1.plot([x, x + ez_global[0]],
                #         [y, y + ez_global[1]],
                #         [z, z + ez_global[2]],
                #         color='orange', linewidth=2)
                    
             
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Z')
            ax1.set_title(str('Trajectory (3D Ansicht) MSE: ' + str(sum(self.mseHistory))))

            # 2) Oberer, rechter Subplot: 3D-Ansicht aus Y-Richtung
            ax2 = fig.add_subplot(gs[0, 1], projection='3d')
            ax2.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                    color='red', marker='o', linestyle='-')
            ax2.plot(optimTraj[:, 0], optimTraj[:, 1], optimTraj[:, 2],
                    color='green', marker='o', linestyle='-')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_zlabel('Z')
            ax2.set_title('Seitliche Ansicht (Y-Richtung)')
            ax2.view_init(elev=10, azim=140)

            # 3) Unterer Subplot (über die gesamte Breite): 2D-Plot für den Reward
            ax3 = fig.add_subplot(gs[1, :])
            # Hier plotten wir den Reward-Verlauf:
            ax3.plot(range(len(self.rewardHistory)), self.rewardHistory, marker='o', color='blue')
            ax3.set_xlabel('Step')
            ax3.set_ylabel('Reward')
            ax3.set_title('Reward vs. Step')
            ax3.grid(True)

                        # 3) Unterer Subplot (über die gesamte Breite): 2D-Plot für den Reward
            ax4 = fig.add_subplot(gs[2, :])
            # Hier plotten wir den Reward-Verlauf:
            #ax4.plot(range(len(angles[:,0])), angles[:,0], marker='o', color='red')
            ax4.plot(range(len(angles[:,1])), angles[:,1], marker='o', color='orange')
            #ax4.plot(range(len(angles[:,2])), angles[:,2], marker='o', color='green')
            ax4.set_xlabel('Step')
            ax4.set_ylabel('Angle')
            ax4.set_title('Angle')
            ax4.grid(True)

            now = datetime.datetime.now()

            #Datum/Zeit als String formatieren (z.B. "2025-03-13_15-30-45")
            timestamp_str = now.strftime("%Y-%m-%d%H-%M-%S")

            #Dateiname mit Zeitstempel
            filename += f"/Plot_{timestamp_str}.png"

            plt.tight_layout()
            plt.savefig(filename)
            plt.close(fig)
        except:
            pass


# ---------------------------------- TESTING ---------------------------------

def main():
    
    dh_params = [
        [0, 0.15185, 0, 90],
        [0, 0, -0.24355, 0],
        [0, 0, -0.2132, 0],
        [0, 0.13105, 0, 90],
        [0, 0.08535, 0, -90],
        [0, 0.0921, 0, 0]
    ]

    joint_angles = [0, -90, 90, 90, 90, 0]


    env = Env(dh_params, joint_angles)
    print("x")

    env.reset()

    traj = []

    for i in range(10):
        test_traj, _, _, _, _ = env.step([-0.1, 0.1, 0.1, 0, 0.1, 0])
        test_traj = env.robot.get_tcp_pose(test_traj)
        print(test_traj)
        traj.append(test_traj)

    env.render_trajectory(traj, "testfig.png")

    # for i in range(100):
    #     env.step(600)

    #env.checkBounds()


if __name__ == "__main__":
    main()