import os
import numpy as np
import scipy.integrate as integrate
from scipy.interpolate import UnivariateSpline, make_interp_spline, splrep, BSpline
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import cvxpy as cp
import networkx as nx

def matrix_ODE(t,Y,C):
    """
    ----------
    Computes the ODE of a Chemical Mechanism using its Coefficient Matrix
    ----------
    Parameters 
    ----------
    Y  : float array - vector of species' concentrations
    C  : float array - coefficient matrix
    """
    s = len(Y)
    dvec = np.zeros(s)
    for i in range(s):
        for j in range(i+1):
            Y = np.hstack((Y,Y[i]*Y[j]))
    #Y = np.hstack((np.ones(1),Y))
    for k in range(s):
        dvec[k] = C[k]@Y.T        
    return dvec

def build_dictionary(labels, data):
    """
    ----------
    Builds the quadratic dictionary matrix and the labels of each row
    ----------
    Parameters 
    ----------
    labels  : str array   - labels of model species
    data    : float array - data set of solutions
    ----------
    Returns
    ----------
    dictionary  : float array - dictionary matrix
    dict_labels : str array   - labels of dictionary functions 
    """
    dictionary  = data.copy()
    dict_labels = labels.copy()
    s           = len(labels)
    for i in range(s):
        for j in range(i+1):
            yi = data[i,:].reshape(1,-1)
            yj = data[j,:].reshape(1,-1)
            
            dictionary = np.vstack((dictionary,yi*yj))
            dict_labels.append(labels[i]+"*"+labels[j])
    
    # Add constant biased vector to the dictionary for constant term in the ODE system
    #dict_labels.insert(0,'1')
    #dictionary = np.vstack([np.ones((1, dictionary[0].shape[0])),dictionary])

    return dictionary, dict_labels            

def build_int_dif_matrix(t_eval, w, spline_deg = 3, spline_type = 'not-a-knot'):
    """
    ----------
    Builds the cumulative integration matrix and differention matrix
    ----------
    Parameters 
    ----------
    npts        : int         - number of data points
    t_eval      : float array - time points 
    w           : int         - number of experiments
    spline_deg  : int         - integration matrix spline degree (default 3)
    spline_type : int         - integration matrix spline type (default=not-a-knot)
    """
    npts = len(t_eval)
    
    int_matrix = np.zeros((npts,npts))
    dif_matrix = np.zeros((npts,npts))
    
    for i in range(npts):
        y = np.zeros(npts)
        y[i] = 1.0
        spl = make_interp_spline(t_eval, y, k=spline_deg, bc_type=spline_type)
        int_matrix[i,:] = spl.antiderivative()(t_eval) - spl.antiderivative()(t_eval[0])
        dif_matrix[i,:] = spl.derivative()(t_eval)

    K = np.kron(np.eye(w),int_matrix)
    L = np.kron(np.eye(w),dif_matrix)
    return K, L

def build_ivp_matrix(npts, X):
    """    
    ----------
    Builds the initial values matrix
    ----------
    Parameters 
    ----------
    npts  : int         - number of data points
    X     : float array - data matrix
    ----------
    Returns
    ----------
    ivp_matrix : float array - initial values matrix
    """
    M = X.shape[0]
    w = X.shape[1] // npts
    ivp_matrix = np.zeros((M,0))
    for j in range(w):
        tind = j*npts
        x0 = X[:,tind]
        ivp = x0.reshape(-1,1)@np.ones((1,npts)) 
        ivp_matrix = np.hstack((ivp_matrix, ivp)) 
    return ivp_matrix

def itls(D, Y, threshold=0.5):
    """
    ----------
    Iterative Thresholded Least Squares
    ----------
    Parameters 
    ----------
    D         : float array - dictionary matrix (For Integral SINDy, D -> D@K)
    Y         : float array - data matrix (For Integral SINDy, Y -> Y-IVP)
    threshold : float       - threshold parameter (tau)
    ----------
    Returns
    ----------
    C         : float array - sparse coefficient matrix
    """
    s, d = Y.shape[0], D.shape[0]
    C = np.zeros((s, d))

    for i in range(s):
        # Initialize mask and keep indices
        keep = np.arange(d)
        X_curr = D.T.copy()
        y = Y[i, :]

        while True:
            # Least squares on current subset
            w_ls, _, _, _ = np.linalg.lstsq(X_curr[:, keep], y, rcond=None)

            # Thresholding
            k = np.abs(w_ls) >= threshold
            new_keep = keep[k]

            if np.array_equal(new_keep, keep):  # Convergence criterion
                break

            keep = new_keep

        # Store result in full vector
        w = np.zeros(d)
        w[keep] = w_ls
        C[i, :] = w
        
    return C

def remove_zero_columns(C):
    """
    ----------
    Removes columns that contain only zeros from a matrix and returns the filtered matrix
    along with the indices of the remaining columns.
    ----------
    Parameters
    ----------
    C : float array - matrix to be reduced
    ----------
    Returns
    ----------
    C_red             : float array - matrix with zero-only columns removed
    remaining_indices : int array   - array of indexes of active columns after filtering
    """
    # Find nonzero columns
    nonzero_columns = np.any(C != 0, axis=0)
    remaining_indices = np.where(nonzero_columns)[0]
    
    # Select only nonzero columns
    return C[:, nonzero_columns], remaining_indices

def generate_stoichiometry_matrix(M,N):
    """
    ----------
    Generates a stoichiometry matrix S for n variables including pairwise combinations and a constant term.
    ----------
    Parameters
    ----------
    M : int - Number of features
    N : int - Number of quadratic combinations of features
    ----------
    Returns
    ----------
    S : int array - Stoichiometry matrix of shape (n, (n+1)(n+2)/2).
    """
    variables = [f"x{i+1}" for i in range(M)]
    
    # Generate pairwise combinations
    for i in range(M):
        for j in range(i + 1):
            variables.append(variables[i] + "*" + variables[j])
    
    # Insert constant term at the beginning
    #variables.insert(0, '1')
    
    # Initialize stoichiometry matrix
    S = np.zeros((M, N), dtype=int)
    
    # Fill matrix
    for col, var in enumerate(variables):
        #if var == '1':
            #continue  # The constant term has no active rows
        terms = var.split('*')
        for term in terms:
            row = int(term[1:]) - 1  # Extract index from 'xN'
            S[row, col] += 1  # Increment by 1, add 2 if it's a squared term

    return S
                 
def reduce_model(C,M,N,D_labels):
    """
    Reduces the model for Kirchhoff matrix computation
    
    Parameters:
    C (np.ndarray): A NumPy array. (The coefficient matrix found by SINDy)
    s (int): Number of species
    
    Returns:
    S (np.ndarray): The reduced stoichiometry matrix of the identified system
    C_red (np.ndarray): The reduced coefficient matrix of the identified system
    
    """
                 
    C_red, indx = remove_zero_columns(C)
    S = generate_stoichiometry_matrix(M,N)
                 
    S_red      = np.zeros((M,len(indx)))
    labels_red = []
    for i in range(len(indx)):
        S_red[:,i] = S[:,indx[i]]
        labels_red.append(D_labels[indx[i]])
    
    return C_red, S_red, labels_red


def reduce_model2(C, M, N, D_labels):
    """
    Reduces the model for Kirchhoff matrix computation.
    Always keeps the first M columns (linear monomials), and filters remaining columns (quadratic monomials onwards)
    by removing zero-only ones. Updates D_labels and stoichiometry matrix accordingly.
    """
    # Split the matrix
    C_fixed = C[:, :M]     # always keep
    C_rest  = C[:, M:]     # subject to filtering

    # Filter the remaining part
    C_rest_red, rest_indices = remove_zero_columns(C_rest)

    # Combine fixed and reduced parts
    C_red = np.hstack((C_fixed, C_rest_red))

    # Build full stoichiometry matrix
    S = generate_stoichiometry_matrix(M, N)

    # Always keep first M columns + filtered remaining columns
    keep_indices = np.concatenate((np.arange(M), M + rest_indices))

    # Reduced stoichiometry
    S_red = S[:, keep_indices]

    # Reduced labels
    labels_red = [D_labels[i] for i in keep_indices]

    return C_red, S_red, labels_red

def support_mismatch(C1, C2):
    return np.sum((C1 != 0) != (C2 != 0))

def solve_kirchhoff(Q, C_red):
    # Dimensions
    num_complexes = Q.shape[1]
    
    # Define K as a variable
    K = cp.Variable((num_complexes, num_complexes))
    
    # Objective: Minimize row-wise least squares error min_{K} ||QK - C||_{F}
    objective = cp.Minimize(cp.norm(Q @ K - C_red, 'fro'))

    # === Constraints ===
    constraints = []
    
    # Kirchhoff: column sums must be zero
    for j in range(num_complexes):
        constraints.append(cp.sum(K[:, j]) == 0)

    # Non-negativity for off-diagonal elements
    for i in range(num_complexes):
        for j in range(num_complexes):
            if i != j:
                constraints.append(K[i, j] >= 0)
    
    # Non-positivity for diagonal elements
    for i in range(num_complexes):
        constraints.append(K[i, i] <= 0)
    
    # Solve the optimization problem
    problem = cp.Problem(objective, constraints)
    problem.solve()
    
    return K.value

def draw_directed_graph_from_kirchhoff(K, labels=None, title="Graph", folder_path="Graph_folder", 
                                       positions=None, node_colours=None, size=(10, 1), threshold=1e-10):
    """
    Draws a directed graph from a given Kirchhoff (Laplacian) matrix with custom vertex labels.
    
    K: numpy array representing the Kirchhoff matrix
    labels: list of node labels, default is numeric indices
    title: Title of the plot
    positions: Dictionary of node positions {label: (x, y)}, optional
    size: Tuple for figure size
    threshold: Minimum absolute weight to consider an edge relevant
    """
    if K.shape[0] != K.shape[1]:
        raise ValueError("Kirchhoff matrix must be square.")
    
    num_nodes = K.shape[0]
    if labels is None:
        labels = list(range(num_nodes))
    elif len(labels) != num_nodes:
        raise ValueError("Labels array must have the same length as the number of nodes.")
    
    # Compute adjacency matrix from Kirchhoff
    A = -K.copy()
    np.fill_diagonal(A, 0)
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add only relevant edges (weight > threshold)
    for i in range(num_nodes):
        for j in range(num_nodes):
            weight = A[i, j]
            if weight > threshold:
                G.add_edge(labels[i], labels[j], weight=weight)
    
    # Automatically add only the nodes involved in edges
    used_labels = set([u for edge in G.edges() for u in edge])
    
    # Use custom positions if provided, filtered by used labels
    if positions is not None:
        pos = {label: coord for label, coord in positions.items() if label in used_labels}
    else:
        pos = nx.circular_layout(G)

    plt.figure(figsize=size)
    
    nx.draw(
        G,
        pos,
        with_labels=True,
        labels={node: node for node in G.nodes()},
        node_color=[node_colours[node] for node in G.nodes()],
        edge_color='gray',
        node_size=2000,
        font_size=12,
        font_color='white',
        arrows=True
    )
    
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    edge_labels = {}
    for i, j in G.edges():
        weight = -K[label_to_index[i], label_to_index[j]]
        if abs(weight) > threshold:
            edge_labels[(i, j)] = f"{np.round(weight, 4)} ( {i} → {j} )"
    
    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_size=10,
        label_pos=0.3,
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
    )

    #plt.axis('equal') 
    plt.title(title)
    savefig_path = os.path.join(folder_path, title)
    plt.savefig(savefig_path + ".pdf", transparent=False, bbox_inches='tight')
    #plt.savefig(title + ".pdf", transparent=False, bbox_inches='tight')
    plt.show()



