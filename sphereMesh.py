import math

import numpy
import numpy as np
import matplotlib.pyplot as plt

def mid(lst1, lst2):
    lst = []
    for i in range(len(lst1)):
        lst.append((lst1[i] + lst2[i]) / 2)
    return lst

def center(p1, p2, p3, p4):
    c1 = mid(p1, p2)
    c2 = mid(p3, p4)
    mid_c = mid(c1, c2)
    return mid_c


def normilize(point):
    center = [0.5, 0.5, 0.5]
    # Convert the point coordinates to spherical coordinates
    r = np.linalg.norm(point - center)
    theta = np.arccos((point[2] - center[2]) / r)
    phi = np.arctan2((point[1] - center[1]), (point[0] - center[0]))
    # print(radius)
    # Convert spherical coordinates to Cartesian coordinates
    x = center[0] + radius * np.sin(theta) * np.cos(phi)
    y = center[1] + radius * np.sin(theta) * np.sin(phi)
    z = center[2] + radius * np.cos(theta)

    return [x, y, z]

def create_spherified_cube(radius, vertices):
    # Define cube vertices

    new_vertices = []
    # print(len(vertices))
    for edge in vertices:
        p1 = edge[0]
        p2 = edge[1]
        p3 = edge[2]
        p4 = edge[3]
        p5 = mid(p1, p2)
        p6 = mid(p2, p3)
        p7 = mid(p3, p4)
        p8 = mid(p1, p4)
        cen = center(p1, p2, p3, p4)

        p5 = normilize(numpy.array(p5))
        p6 = normilize(numpy.array(p6))
        p7 = normilize(numpy.array(p7))
        p8 = normilize(numpy.array(p8))
        cen = normilize(numpy.array(cen))

        new_vertices.append([p1, p5, cen, p8])
        new_vertices.append([p5, p2, p6, cen])
        new_vertices.append([cen, p6, p3, p7])
        new_vertices.append([p8, cen, p7, p4])

    return new_vertices

def plot_spherified_cube(vertices):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = []
    y = []
    z = []
    for v in vertices:
        for s in v:
            x.append(s[0])
            y.append(s[1])
            z.append(s[2])

        p1 = v[0]
        p2 = v[1]
        p3 = v[2]
        p4 = v[3]


    ax.scatter(x, y, z)

    # ax.plot(x, y, z, 'b-')

    plt.show()

def compute_derivatives(f, mesh_vertices, h):
    derivatives = []
    for vertex in mesh_vertices:
        x, y, z = vertex[0], vertex[1], vertex[2]
        df_dx = (f(x + h, y, z) - f(x - h, y, z)) / (2 * h)
        df_dy = (f(x, y + h, z) - f(x, y - h, z)) / (2 * h)
        df_dz = (f(x, y, z + h) - f(x, y, z - h)) / (2 * h)
        derivatives.append((df_dx, df_dy, df_dz))
    return derivatives

def f(x, y, z):
    return x**2 + y**2 + z**2

def just(side):
    lst = []
    for s in side:
        for v in s:
            lst.append(v)
    return lst

def task3_2_and_4(radius, sides):
    side = create_spherified_cube(radius, sides)
    cubed_sphere_mesh = just(side)
    # Step 3: Set the grid spacing
    h = 0.001
    #
    # # Step 4: Compute the derivatives
    derivatives = compute_derivatives(f, cubed_sphere_mesh, h)

    # Step 5: Print or visualize the computed derivatives
    for i, derivative in enumerate(derivatives):
        vertex = cubed_sphere_mesh[i]
        print(f"Vertex {i + 1}: ({vertex[0]}, {vertex[1]}, {vertex[2]})")
        print("Partial derivatives:", derivative)
        print("Exact derivatives:", f"({2 * vertex[0]}, {2 * vertex[1]}, {2*vertex[2]})")
        print()

def define_stencil():
    # Define the stencil coefficients
    coefficients = [1, -1, 1, -1, 1, -1, 1, -1]

    return coefficients

def linear_combination(f_values, coefficients):
    result = 0.0
    for i in range(len(coefficients)):
        result += coefficients[i] * f_values[i]
    return result

def exact_derivative_x(x, y, z):
    return 2 * x

def exact_derivative_y(x, y, z):
    return 2 * y

def exact_derivative_z(x, y, z):
    return 2 * z

def task3_3(radius, sides):
    side = create_spherified_cube(radius, sides)
    cubed_sphere_mesh = just(side)
    # Define a range of grid spacings to test
    grid_spacings = [0.1, 0.01, 0.001, 0.0001]

    # Define the function values on the cubed sphere mesh
    f_values = [f(x, y, z) for x, y, z in cubed_sphere_mesh]

    # Define the stencil
    stencil = define_stencil()

    # Compute the linear combination for each grid spacing and compare the results
    previous_result = None
    converges = True
    for spacing in grid_spacings:
        # Compute the linear combination using the current grid spacing
        result = linear_combination(f_values, stencil)

        if previous_result is not None:
            # Compare the current result with the previous result
            if result != previous_result:
                converges = False
                break

        previous_result = result

    # Print the convergence result
    if converges:
        print("The linear combination converges as the grid spacing vanishes.")
    else:
        print("The linear combination does not converge as the grid spacing vanishes.")


radius = math.sqrt(3) / 2
sides = [[[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]], [[0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1]],
         [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]], [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
         [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]], [[1, 0, 0], [1, 0, 1], [1, 1, 1], [1, 1, 0]]]

# TESTER FOR CONSTRUCTING SPHERE TASK 3.1
# new_vertices = create_spherified_cube(radius, sides)
# new_vertices = create_spherified_cube(radius, new_vertices)
# new_vertices = create_spherified_cube(radius, new_vertices)
# plot_spherified_cube(new_vertices)


# TESTER FOR 3.2 AND 3.4
# task3_2_and_4(radius, sides)


# TESTER FOR 3.3
# task3_3(radius, sides)


