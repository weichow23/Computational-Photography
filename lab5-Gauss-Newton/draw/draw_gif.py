import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def read_coordinates(filename):
    coordinates = []
    with open(filename, 'r') as file:
        for line in file:
            coords = list(map(float, line.strip().split()))
            coordinates.append(coords)
    return np.array(coordinates)

def calculate_ellipsoid_data(coordinates, A, B, C):
    center = np.mean(coordinates, axis=0)
    a = np.sqrt(A)
    b = np.sqrt(B)
    c = np.sqrt(C)
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x_ellipsoid = center[0] + a * np.outer(np.cos(u), np.sin(v))
    y_ellipsoid = center[1] + b * np.outer(np.sin(u), np.sin(v))
    z_ellipsoid = center[2] + c * np.outer(np.ones_like(u), np.cos(v))
    return x_ellipsoid, y_ellipsoid, z_ellipsoid

def update_plot(frame, ax, coordinates):
    ax.view_init(azim=frame)

def main():
    filename = 'ellipse753.txt'
    coordinates = read_coordinates(filename)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], c='b', marker='o', label='Coordinates')

    A = 2.94404
    B = 2.30504
    C = 1.79783
    x_ellipsoid, y_ellipsoid, z_ellipsoid = calculate_ellipsoid_data(coordinates, A**2, B**2, C**2)
    ax.plot_surface(x_ellipsoid, y_ellipsoid, z_ellipsoid, color='r', alpha=0.2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Scatter Plot and Ellipsoid')

    anim = FuncAnimation(fig, update_plot, frames=np.arange(0, 360, 2), fargs=(ax, coordinates), interval=50)
    anim.save('plot.gif')
    plt.show()

if __name__ == "__main__":
    main()
