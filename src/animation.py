import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import imageio


class Animation():
    def __init__(self, width=800, height=600, save_gif=False):

        self.width = width
        self.height = height
        self.save_fig = save_gif

        self.cube_vertices = np.array(
            [
                [-1, -1, -1],
                [ 1, -1, -1],
                [ 1,  1, -1],
                [-1,  1, -1],
                [-1, -1,  1],
                [ 1, -1,  1],
                [ 1,  1,  1],
                [-1,  1,  1]
            ]
        )

        self.cube_edges = np.array(
            [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 0],
                [4, 5],
                [5, 6],
                [6, 7],
                [7, 4],
                [0, 4],
                [1, 5],
                [2, 6],
                [3, 7]
            ]
        )

        self.cube_colors = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 1, 0],
                [1, 0, 1],
                [0, 1, 1]
            ]
        )

    def _render_cube(self, rotation_matrix):

        rotated_vertices = np.dot(self.cube_vertices, rotation_matrix)
        
        glLineWidth(8.0)
        glBegin(GL_LINES)
        for edge in self.cube_edges:
            for vertex in edge:
                glColor3fv(self.cube_colors[vertex % 6])
                glVertex3fv(rotated_vertices[vertex])
        glEnd()

    def _render_fixed_rf(self, rotation_matrix):

        rf_x = np.dot(np.array([1, 0, 0]), rotation_matrix)
        rf_y = np.dot(np.array([0, 1, 0]), rotation_matrix)
        rf_z = np.dot(np.array([0, 0, 1]), rotation_matrix)

        glLineWidth(1.0)  # Set the line thickness to 1.0
        glBegin(GL_LINES)
        # Draw x-axis in red
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(0, 0, 0)
        glVertex3f(rf_x[0], rf_x[1], rf_x[2])
        # Draw y-axis in green
        glColor3f(0.0, 1.0, 0.0)
        glVertex3f(0, 0, 0)
        glVertex3f(rf_y[0], rf_y[1], rf_y[2])
        # Draw z-axis in blue
        glColor3f(0.0, 0.0, 1.0)
        glVertex3f(0, 0, 0)
        glVertex3f(rf_z[0], rf_z[1], rf_z[2])
        glEnd()

    def _render_target_rf(self, rotation_matrix):

        rf_x = np.dot(np.array([1, 0, 0]), rotation_matrix)
        rf_y = np.dot(np.array([0, 1, 0]), rotation_matrix)
        rf_z = np.dot(np.array([0, 0, 1]), rotation_matrix)

        glLineWidth(1.0)  # Set the line thickness to 1.0
        glBegin(GL_LINES)
        # Draw x-axis in red
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(0, 0, 0)
        glVertex3f(rf_x[0], rf_x[1], rf_x[2])
        # Draw y-axis in green
        glColor3f(0.0, 1.0, 0.0)
        glVertex3f(0, 0, 0)
        glVertex3f(rf_y[0], rf_y[1], rf_y[2])
        # Draw z-axis in blue
        glColor3f(0.0, 0.0, 1.0)
        glVertex3f(0, 0, 0)
        glVertex3f(rf_z[0], rf_z[1], rf_z[2])
        glEnd()

    def animate(self, quaternions_list, target_quaternion, time_step):

        pygame.init()
        pygame.display.set_mode((self.width, self.height), DOUBLEBUF|OPENGL)

        # Initialize OpenGL
        glViewport(0, 0, self.width, self.height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60, self.width / self.height, 0.1, 50.0)
        # gluPerspective(45, self.width / self.height, 0.1, 50.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(0.5, 2, 3, 0, 0, 0, 0, 0, 1)
        # gluLookAt(3, 3, 3, 0, 0, 0, 0, 0, 1)
        glEnable(GL_DEPTH_TEST)


        frames = list()

        q0, q1, q2, q3 = target_quaternion
        target_rotation_matrix = np.array(
            [
                [1 - 2 * q2**2 - 2 * q3**2, 2 * q1 * q2 - 2 * q0 * q3, 2 * q1 * q3 + 2 * q0 * q2],
                [2 * q1 * q2 + 2 * q0 * q3, 1 - 2 * q1**2 - 2 * q3**2, 2 * q2 * q3 - 2 * q0 * q1],
                [2 * q1 * q3 - 2 * q0 * q2, 2 * q2 * q3 + 2 * q0 * q1, 1 - 2 * q1**2 - 2 * q2**2]
            ],
            dtype=np.float32
        )

        for quaternion in quaternions_list:

            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    quit()

            q0, q1, q2, q3 = quaternion
            rotation_matrix = np.array(
                [
                    [1 - 2 * q2**2 - 2 * q3**2, 2 * q1 * q2 - 2 * q0 * q3, 2 * q1 * q3 + 2 * q0 * q2],
                    [2 * q1 * q2 + 2 * q0 * q3, 1 - 2 * q1**2 - 2 * q3**2, 2 * q2 * q3 - 2 * q0 * q1],
                    [2 * q1 * q3 - 2 * q0 * q2, 2 * q2 * q3 + 2 * q0 * q1, 1 - 2 * q1**2 - 2 * q2**2]
                ],
                dtype=np.float32
            )

            self._render_target_rf(target_rotation_matrix)
            self._render_fixed_rf(rotation_matrix)
            self._render_cube(rotation_matrix)

            pygame.display.flip()
            pygame.time.wait(int(time_step * 1000))

            buffer = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
            image = np.frombuffer(buffer, dtype=np.uint8).reshape(self.height, self.width, 3)
            image = np.flipud(image)
            frames.append(image)

            glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

        pygame.quit()

        if self.save_fig:
            imageio.mimsave('cube_rotation.gif', frames, fps=1/time_step)

