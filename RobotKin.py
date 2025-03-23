import numpy as np
import matplotlib.pyplot as plt

class RobotArm:
    def __init__(self, dh_matrix, init):
        self.dh_matrix = dh_matrix
        self.init = init

    def Fdh_matrix(self, theta, d, a, alpha):
        """Erstellt die Denavit-Hartenberg-Transformationsmatrix."""
        theta = np.radians(theta)  # Umwandlung in Radiant
        alpha = np.radians(alpha)
        
        return np.array([
            [np.cos(theta), -np.sin(theta) * np.cos(alpha),  np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
            [np.sin(theta),  np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
            [0,             np.sin(alpha),                 np.cos(alpha),                 d],
            [0,             0,                            0,                            1]
        ])
    

    def rotation_matrix_to_euler_angles(self, R):
        """Berechnet die Euler-Winkel (ZYX-Konvention) aus einer Rotationsmatrix."""
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
        
        return np.degrees([x, y, z])

    def euler_to_rotation_matrix(self, alpha, beta, gamma):
        """
        Wandelt Euler-Winkel (alpha, beta, gamma) in Grad in eine
        3x3-Rotationsmatrix um. Hier wird die Reihenfolge (Rz * Ry * Rx) angenommen.
        Passen Sie das ggf. an Ihre eigene Konvention an.
        """
        # Umrechnung von Grad in Radiant
        alpha_r = np.radians(alpha)
        beta_r  = np.radians(beta)
        gamma_r = np.radians(gamma)

        # Rotationen um x, y, z (Roll, Pitch, Yaw)
        Rx = np.array([
            [1,              0,               0],
            [0,  np.cos(alpha_r), -np.sin(alpha_r)],
            [0,  np.sin(alpha_r),  np.cos(alpha_r)]
        ])

        Ry = np.array([
            [ np.cos(beta_r), 0, np.sin(beta_r)],
            [              0, 1,             0],
            [-np.sin(beta_r), 0, np.cos(beta_r)]
        ])

        Rz = np.array([
            [np.cos(gamma_r), -np.sin(gamma_r), 0],
            [np.sin(gamma_r),  np.cos(gamma_r), 0],
            [              0,                0, 1]
        ])

        # Reihenfolge: Rz * Ry * Rx
        R = Rz @ Ry @ Rx
        return R

    def forward_kinematics(self, joint_angles):
        """Berechnet die Positionen !ALLER! Gelenke und den Endeffektor."""
        T = np.eye(4)
        joint_positions = [(0, 0, 0)]  # Startposition des ersten Gelenks
        
        for i in range(len(self.dh_matrix)):
            theta, d, a, alpha = self.dh_matrix[i]
            T = np.dot(T, self.Fdh_matrix(joint_angles[i] + theta, d, a, alpha))
            joint_positions.append(tuple(T[:3, 3]))
        
        euler_angles = self.rotation_matrix_to_euler_angles(T[:3, :3])
        return joint_positions, euler_angles
    

    def plot(self, joint_angles, save=False, filename="trajectory.png", action=""):
        """Plottet den Roboterarm in 3D."""

        joint_positions, angles = self.forward_kinematics(joint_angles)

        fig = plt.figure(figsize=(8, 6), dpi=150)
        ax = fig.add_subplot(111, projection='3d')
        
        x_vals, y_vals, z_vals = zip(*joint_positions)
        ax.plot(x_vals, y_vals, z_vals, marker='o', linestyle='-', markersize=8, label='Roboterarm')
        
        ax.set_xlabel("X-Achse [m]")
        ax.set_ylabel("Y-Achse [m]")
        ax.set_zlabel("Z-Achse [m]")
        #ax.set_title("3D-Darstellung des Roboterarms")
        ax.set_title("Action: " + action)

        axis_length = 0.05  # Länge der Achsen
        #for i in range(len(angles)):
        alpha, beta, gamma = angles

        # Rotationsmatrix aus Euler-Winkeln
        R = self.euler_to_rotation_matrix(alpha, beta, gamma)

        # Lokale Einheitsvektoren für x, y, z (im Endeffektor-KS)
        ex_local = np.array([1, 0, 0])
        ey_local = np.array([0, 1, 0])
        ez_local = np.array([0, 0, 1])

        # In den Welt-Raum transformieren und auf axis_length skalieren
        ex_global = R @ ex_local * axis_length
        ey_global = R @ ey_local * axis_length
        ez_global = R @ ez_local * axis_length

        # Ursprung jeder kleinen Achse ist (x, y, z).
        # Zeichne drei Linien: (x->x+ex_global), (y->y+ey_global) usw.
        ptIndex = 6
        ax.plot([x_vals[ptIndex], x_vals[ptIndex] + ex_global[0]],
                [y_vals[ptIndex], y_vals[ptIndex] + ex_global[1]],
                [z_vals[ptIndex], z_vals[ptIndex] + ex_global[2]],
                color='blue', linewidth=2)

        ax.plot([x_vals[ptIndex], x_vals[ptIndex] + ey_global[0]],
                [y_vals[ptIndex], y_vals[ptIndex] + ey_global[1]],
                [z_vals[ptIndex], z_vals[ptIndex] + ey_global[2]],
                color='green', linewidth=2)

        ax.plot([x_vals[ptIndex], x_vals[ptIndex] + ez_global[0]],
                [y_vals[ptIndex], y_vals[ptIndex] + ez_global[1]],
                [z_vals[ptIndex], z_vals[ptIndex] + ez_global[2]],
                color='orange', linewidth=2)

        if save:
            plt.savefig(filename)
            plt.close(fig)
    

    def get_tcp_pose(self, joint_angles):
        """Berechnet die Endeffektor-Position mittels DH-Parametern."""
        T = np.eye(4)
        for i in range(6):
            theta, d, a, alpha = self.dh_matrix[i]
            T = np.dot(T, self.Fdh_matrix(joint_angles[i] + theta, d, a, alpha))
        
        x, y, z = T[:3, 3]
        euler_angles = self.rotation_matrix_to_euler_angles(T[:3, :3])
        return x, y, z, euler_angles[0], euler_angles[1], euler_angles[2]
    


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

    joint_angles = [180, -90, 90, 90, 90, -45]
    
    robot = RobotArm(dh_params, joint_angles)
    robot.plot(robot.init)
    joint_angles = [180, -90, 90, 90, 90, 45]
    robot.plot(joint_angles)

    x, y, z, alpha, beta, gamma = robot.get_tcp_pose(joint_angles)

    print(f"\nEndeffektor Position: x={x:.3f}, y={y:.3f}, z={z:.3f}, alpha={alpha:.3f}, beta={beta:.3f}, gamma={gamma:.3f}")


    plt.show()

if __name__ == "__main__":
    main()