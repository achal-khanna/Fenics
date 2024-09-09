# Materials decided - Grade 5 Ti-6Al-4V Alloy (UNS R56200), Aluminum 6061 Alloy (UNS A96061)
# https://www.azom.com/article.aspx?ArticleID=6636
# https://www.azom.com/article.aspx?ArticleID=9299

# Taken inspiration and referenced from Mechanical MNIST - https://github.com/elejeune11/Mechanical-MNIST/blob/master/generate_dataset/Uniaxial_Extension_FEA_train_FEniCS.py
# Referred from FEniCS guides - https://comet-fenics.readthedocs.io/en/latest/demo/elasticity/2D_elasticity.py.html, https://fenics-solid-tutorial.readthedocs.io/en/latest/2DNonlinearElasticity/2DNonlinearElasticity.html

# Units are in mm, MPa

import numpy as np
import os
import matplotlib.pyplot as plt
import idx2numpy
import fenics as fe
import time

def bitmap(x, y, data):
    """Generate bitmap based on input coordinates and MNIST image data."""
    total = fe.Constant(0.0)
    for jj in range(data.shape[0]):
        for kk in range(data.shape[1]):
            const1 = fe.conditional(x >= fe.Constant(jj), fe.Constant(1), fe.Constant(0))
            const2 = fe.conditional(x < fe.Constant(jj + 1), fe.Constant(1), fe.Constant(0))
            const3 = fe.conditional(y >= fe.Constant(kk), fe.Constant(1), fe.Constant(0))
            const4 = fe.conditional(y < fe.Constant(kk + 1), fe.Constant(1), fe.Constant(0))
            sum_ = const1 + const2 + const3 + const4
            const = fe.conditional(sum_ > fe.Constant(3), fe.Constant(1), fe.Constant(0))
            total += const * fe.Constant(float(data[jj, kk]))
    return total

class GetMat:
    def __init__(self, mesh, youngs_min, youngs_max, poissons_ratio):
        self.mesh = mesh
        self.youngs_min = youngs_min
        self.youngs_max = youngs_max
        self.nu = poissons_ratio
        
    def getFunctionMaterials(self, data):
        x = fe.SpatialCoordinate(self.mesh)
        val = bitmap(x[0], x[1], data)
        E = val / 255.0 * (self.youngs_max - self.youngs_min) + self.youngs_min
        return E, self.nu

def epsilon(u):
    return fe.sym(fe.grad(u))

def sigma(u, lmbda, mu):
    return lmbda*fe.div(u)*fe.Identity(2) + 2*mu*epsilon(u)

def pix_centers(E, mesh):
    Exx = np.zeros((28, 28))
    Exy = np.zeros((28, 28))
    Eyy = np.zeros((28, 28))

    # Project E onto a TensorFunctionSpace for interpolation
    V = fe.TensorFunctionSpace(mesh, 'CG', 1)
    E_tensor = fe.project(E, V)

    # Iterate through each point on the grid
    for kk in range(28):
        for jj in range(28):
            xx = jj + 0.5 
            yy = kk + 0.5
         
            # Evaluate components of E at the interpolated function
            E_eval = E_tensor(xx, yy)  
            Exx[kk, jj] = E_eval[0]
            Exy[kk, jj] = E_eval[1]
            Eyy[kk, jj] = E_eval[3]

    return Exx, Eyy, Exy

# Set current working directory and create output folder
cwd = os.path.dirname(os.path.abspath(__file__))
output_folder_name = os.path.join(cwd, "Testing")
os.makedirs(output_folder_name, exist_ok=True)

# Load MNIST dataset
train_images_path = os.path.join(cwd, "t10k-images.idx3-ubyte")
train_labels_path = os.path.join(cwd, "t10k-labels.idx1-ubyte")
mnist_image_array = idx2numpy.convert_from_file(train_images_path)
mnist_image_labels = idx2numpy.convert_from_file(train_labels_path)

# Compiler settings / optimization options
fe.parameters["form_compiler"]["cpp_optimize"] = True
fe.parameters["form_compiler"]["representation"] = "uflacs"
fe.parameters["form_compiler"]["quadrature_degree"] = 2

# Mesh and material parameters
mesh_size = 3
p_1_x = 0.0; p_1_y = 0.0;
p_2_x = 28.0; p_2_y = 28.0;
youngs_min = 68900.0
youngs_max = 110000.0
poissons_ratio = 0.32
traction_force = 100.0;
dimension = 28

Exx_combined = []
Exy_combined = []
Eyy_combined = []
Sxx_combined = []
Sxy_combined = []
Syy_combined = []

start_time = time.time()

# Loop over each MNIST image
# for index in range(mnist_image_array.shape[0]):
for index in range(1000, 2000):
    mnist_image_single = mnist_image_array[index]
    mnist_image_label = mnist_image_labels[index]

    # Flip and prepare the data
    data = np.zeros(mnist_image_single.shape)
    for jj in range(data.shape[0]):
        for kk in range(data.shape[1]):
            data[jj,kk] = mnist_image_single[int(27 - kk), jj]

    # Create mesh
    mesh = fe.RectangleMesh(fe.Point(p_1_x, p_1_y), fe.Point(p_2_x, p_2_y), mesh_size * dimension, mesh_size * dimension, "crossed")

    # Define function spaces
    material_function_space = fe.FunctionSpace(mesh, 'Lagrange', 1)
    displacement_function_space = fe.VectorFunctionSpace(mesh, 'Lagrange', 2)
    u_func = fe.Function(displacement_function_space, name = "Displacement")
    u_trial = fe.TrialFunction(displacement_function_space)
    u_test = fe.TestFunction(displacement_function_space)
    
    # Get material properties
    mat = GetMat(mesh, youngs_min, youngs_max, poissons_ratio)
    E, nu = mat.getFunctionMaterials(data)
    
    # Plane strain
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2 * (1 + nu))
    
    # Define the material domain
    matdomain = fe.MeshFunction('size_t', mesh, mesh.topology().dim())
    dx = fe.Measure('dx', domain=mesh, subdomain_data=matdomain)

    # Boundary Condition
    traction_top = fe.Constant((0.0, traction_force))
    body_force = fe.Constant((0.0, 0.0))

    Top = fe.CompiledSubDomain("near(x[1], topCoord)", topCoord = p_2_y)
    Bottom = fe.CompiledSubDomain("near(x[1], btmCoord)", btmCoord = p_1_y)

    boundary_markers = fe.MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
    boundary_markers.set_all(0)  # Initialize all boundaries to 0

    # Mark boundaries using boundary functions
    Top.mark(boundary_markers, 1)
    ds = fe.Measure('ds', domain=mesh, subdomain_data=boundary_markers)

    bcBot = fe.DirichletBC(displacement_function_space.sub(1), fe.Constant(0.0), Bottom)
    bcs = [bcBot]

    d = len(u_func)
    I = fe.Identity(d)             
    a = fe.inner(sigma(u_trial, lmbda, mu), epsilon(u_test))*dx
    L = fe.dot(body_force, u_test)*dx('everywhere') + fe.dot(traction_top, u_test)*ds(1)

    fe.solve(a == L, u_func, bcs)

    E = epsilon(u_func)
    sig = sigma(u_func, lmbda, mu)
    
    Exx, Eyy, Exy = pix_centers(E, mesh)
    Sxx, Syy, Sxy = pix_centers(sig, mesh)

    print(time.time() - start_time)

    fpath = output_folder_name + "/" + str(mnist_image_label) + "/"
    os.makedirs(fpath, exist_ok=True)

    fn_Exx = fpath + str(index) + '_Exx.txt'
    fn_Exy = fpath + str(index) + '_Exy.txt'
    fn_Eyy = fpath + str(index) + '_Eyy.txt'
    fn_Sxx = fpath + str(index) + '_Sxx.txt'
    fn_Sxy = fpath + str(index) + '_Sxy.txt'
    fn_Syy = fpath + str(index) + '_Syy.txt'

    np.savetxt(fn_Exx, Exx)
    np.savetxt(fn_Exy, Exy)
    np.savetxt(fn_Eyy, Eyy)
    np.savetxt(fn_Sxx, Sxx)
    np.savetxt(fn_Sxy, Sxy)
    np.savetxt(fn_Syy, Syy)

    Exx_combined.append(Exx)
    Exy_combined.append(Exy)
    Eyy_combined.append(Eyy)
    Sxx_combined.append(Sxx)
    Sxy_combined.append(Sxy)
    Syy_combined.append(Syy)

n_images = len(Exx_combined)
Exx_combined = np.array(Exx_combined).reshape(n_images, -1)
Exy_combined = np.array(Exy_combined).reshape(n_images, -1)
Eyy_combined = np.array(Eyy_combined).reshape(n_images, -1)
Sxx_combined = np.array(Sxx_combined).reshape(n_images, -1)
Sxy_combined = np.array(Sxy_combined).reshape(n_images, -1)
Syy_combined = np.array(Syy_combined).reshape(n_images, -1)

# Save the arrays
np.savetxt(output_folder_name+ 'Exx_combined.txt', Exx_combined)
np.savetxt(output_folder_name+ 'Exy_combined.txt', Exy_combined)
np.savetxt(output_folder_name+ 'Eyy_combined.txt', Eyy_combined)
np.savetxt(output_folder_name+ 'Sxx_combined.txt', Sxx_combined)
np.savetxt(output_folder_name+ 'Sxy_combined.txt', Sxy_combined)
np.savetxt(output_folder_name+ 'Syy_combined.txt', Syy_combined)

print("Arrays saved successfully.")