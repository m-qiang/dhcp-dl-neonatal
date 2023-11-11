import torch
import numpy as np
from scipy.sparse import coo_matrix


def apply_affine_mat(vert, affine):
    """
    Apply affine transformation to surface vertices.
    
    Inputs:
    - vert: mesh vertices, (|V|,3) numpy.array
    - affine: affine matrix, (4,4) numpy.array
    
    Returns:
    - vert: vertices after affine transform, (|V|,3) numpy.array
    """
    return vert @ affine[:3,:3].T + affine[:3, -1]


def adjacent_faces(face):
    """
    Find the adjacent two faces for each edge
    
    Inputs:
    - face: mesh faces, (1,|F|,3) torch.LongTensor
    
    Returns:
    - adj_faces: indices of two adjacent two faces
    for each edge, (|E|, 2) torch.LongTensor
    
    """
    edge = torch.cat([face[0,:,[0,1]],
                      face[0,:,[1,2]],
                      face[0,:,[2,0]]], axis=0)  # (2|E|, 2)
    nf = face.shape[1]
    # map the edge to its belonging face
    fid = torch.arange(nf).to(face.device)
    adj_faces = torch.cat([fid]*3)  # (3|F|)

    edge = edge.cpu().numpy()
    # sort the edge such that v_i < v_j
    edge = np.sort(edge, axis=-1)
    # sort the edge to find the correspondence 
    # between e_ij and e_ji
    eid = np.lexsort((edge[:,1], edge[:,0]))  # (2|E|)

    # map edge to its adjacent two faces
    adj_faces = adj_faces[eid].reshape(-1,2)  # (|E|, 2)
    return adj_faces


def vert_normal(vert, face):
    """
    Compute the normal vector of each vertex.
    
    This function is retrieved from pytorch3d.
    For original code please see: 
    _compute_vertex_normals function in
    https://pytorch3d.readthedocs.io/en/latest/
    _modules/pytorch3d/structures/meshes.html
    
    Inputs:
    - vert: input mesh vertices, (1,|V|,3) torch.Tensor
    - face: input mesh faces, (1,|F|,3) torch.LongTensor
    
    Returns:
    - v_normal: vertex normals, (1,|V|,3) torch.Tensor
    """
    
    v_normal = torch.zeros_like(vert)   # normals of vertices
    v_f = vert[:, face[0]]   # vertices of each face

    # compute normals of faces
    f_normal_0 = torch.cross(v_f[:,:,1]-v_f[:,:,0], v_f[:,:,2]-v_f[:,:,0], dim=2) 
    f_normal_1 = torch.cross(v_f[:,:,2]-v_f[:,:,1], v_f[:,:,0]-v_f[:,:,1], dim=2) 
    f_normal_2 = torch.cross(v_f[:,:,0]-v_f[:,:,2], v_f[:,:,1]-v_f[:,:,2], dim=2) 

    # sum the faces normals
    v_normal = v_normal.index_add(1, face[0,:,0], f_normal_0)
    v_normal = v_normal.index_add(1, face[0,:,1], f_normal_1)
    v_normal = v_normal.index_add(1, face[0,:,2], f_normal_2)

    v_normal = v_normal / (torch.norm(v_normal, dim=-1).unsqueeze(-1) + 1e-12)
    return v_normal


def face_normal(vert, face):
    """
    Compute the normal vector of each face.
    
    Inputs:
    - vert: input mesh vertices, (1,|V|,3) torch.Tensor
    - face: input mesh faces, (1,|F|,3) torch.LongTensor
    
    Returns:
    - f_normal: face normals, (1,|F|,3) torch.Tensor
    """
    
    v_f = vert[:, face[0]]
    # compute normals of faces
    f_normal = torch.cross(v_f[:,:,1]-v_f[:,:,0],
                           v_f[:,:,2]-v_f[:,:,0], dim=2) 
    f_normal = f_normal / (torch.norm(f_normal, dim=-1).unsqueeze(-1) + 1e-12)

    return f_normal


def mesh_area(vert, face):
    """
    Compute the total area of the mesh

    Inputs:
    - vert: input mesh vertices, (1,|V|,3) torch.Tensor
    - face: input mesh faces, (1,|F|,3) torch.LongTensor
    
    Returns:
    - area: mesh area, float
    """
   
    v0 = vert[:,face[0,:,0]]
    v1 = vert[:,face[0,:,1]]
    v2 = vert[:,face[0,:,2]]
    area = 0.5*torch.norm(torch.cross(v1-v0, v2-v0), dim=-1)
    return area.sum().item()


def face_area(vert, face):
    """
    Compute the area of each face

    Inputs:
    - vert: input mesh vertices, (1,|V|,3) torch.Tensor
    - face: input mesh faces, (1,|F|,3) torch.LongTensor
    
    Returns:
    - area: face area, (|F|,3) torch.Tensor
    """
    
    v0 = vert[:,face[0,:,0]]
    v1 = vert[:,face[0,:,1]]
    v2 = vert[:,face[0,:,2]]
    area = 0.5*torch.norm(torch.cross(v1-v0, v2-v0), dim=-1)
    return area[0]


def adjacency_matrix(face):
    """
    Compute adjacency matrix.
    
    Inputs:
    - face: input mesh faces, (1,|F|,3) torch.LongTensor
    
    Returns:
    - A: adjacency matrix, (1,|V|,|V|) torch.sparse.Tensor
    """

    nv = face.max().item()+1
    edge = torch.cat([face[0,:,[0,1]],
                      face[0,:,[1,2]],
                      face[0,:,[2,0]]], dim=0).T
    # adjacency matrix A
    A = torch.sparse_coo_tensor(
        edge, torch.ones_like(edge[0]).float(), (nv, nv)).unsqueeze(0)
    # number of neighbors for each vertex
    # adj_degree = torch.sparse.sum(A, dim=-1).to_dense().unsqueeze(-1)
    return A #, adj_degree


def neighbor_matrix(face, n_neighbors=2):
    """
    Compute n-neighborhood (n-hop) adjacency matrix.
    
    Inputs:
    - face: input mesh faces, (1,|F|,3) torch.LongTensor
    - n_neighbors: number of hops, int
    
    Returns:
    - A_n: n-hop adjacency matrix, (1,|V|,|V|) torch.sparse.Tensor
    """

    f = face[0].cpu().numpy()
    nv = face.max().item()+1

    # create initial adjacency matrix
    edge = np.concatenate([f[:,[0,1]], f[:,[1,2]], f[:,[2,0]]], axis=0)
    A = coo_matrix((np.ones(edge.shape[0], dtype=np.int8),
                             (edge[:,0], edge[:,1])), shape=(nv, nv))

    # compute connection matrix
    connect_matrix = A
    for n in range(n_neighbors-1):
        connect_matrix = connect_matrix.dot(connect_matrix)
        connect_matrix = (connect_matrix > 0).astype(np.int8).tocoo()
        # remove diagonal elements (self-connection)
        connect_matrix.setdiag(np.zeros(nv))
        connect_matrix.eliminate_zeros()
    
    # create connection matrix An
    edge_n = np.stack([connect_matrix.row, connect_matrix.col])
    edge_n = torch.LongTensor(edge_n).to(face.device)
    A_n = torch.sparse_coo_tensor(
        edge_n, torch.ones_like(edge_n[0]).float(),(nv, nv)).unsqueeze(0)
    return A_n


def laplacian(face):
    """
    Compute Laplacian matrix.
    
    Inputs:
    - face: input mesh faces, (1,|F|,3) torch.LongTensor
    
    Returns:
    - L: Laplacian matrix, (1,|V|,|V|) torch.sparse.Tensor
    """
    nv = face.max().item()+1
    edge = torch.cat([face[0,:,[0,1]],
                      face[0,:,[1,2]],
                      face[0,:,[2,0]]], dim=0).T
    # adjacency matrix A
    A = torch.sparse_coo_tensor(
        edge, torch.ones_like(edge[0]).float(), (nv, nv)).unsqueeze(0)

    # number of neighbors for each vertex
    degree = torch.sparse.sum(A, dim=-1).to_dense()[0]
    weight = 1./degree[edge[0]]
    # normalized adjacency matrix
    A_hat = torch.sparse_coo_tensor(
        edge, weight, (nv, nv)).unsqueeze(0)
    
    # normalized degree matrix, i.e., identity matrix
    # set the diagonal entries to one
    self_edge = torch.arange(nv)[None].repeat([2,1]).to(face.device)
    D_hat = torch.sparse_coo_tensor(
        self_edge, torch.ones_like(self_edge[0]).float(), (nv, nv)).unsqueeze(0)
    L = D_hat - A_hat
    return L


def cot_laplacian(vert, face, eps=1e-12):
    """
    Cotangent Laplacian matrix.
    
    This function is retrieved from pytorch3d.
    For original code please see:
    https://pytorch3d.readthedocs.io/en/latest/_modules/
    pytorch3d/ops/laplacian_matrices.html#cot_laplacian
    
    Inputs:
    - vert: input mesh vertices, (1,|V|,3) torch.Tensor
    - face: input mesh faces, (1,|F|,3) torch.LongTensor
    - eps: small value
    
    Returns:
    - L: cotangent Laplacian matrix, (1,|V|,|V|) torch.sparse.Tensor
    - inv_area: inverse area for each vertex, (1,|V|,1) torch.Tensor
    - degree: degree of each vertex, (1,|V|,1) torch.Tensor
    """
    nv = vert.shape[1]
    nf = face.shape[1]
    v0 = vert[0,face[0,:,0]]
    v1 = vert[0,face[0,:,1]]
    v2 = vert[0,face[0,:,2]]

    # Side lengths of each triangle, of shape (sum(F_n),)
    # A is the side opposite v0, B is opposite v1, and C is opposite v2
    A = (v1 - v2).norm(dim=1)
    B = (v2 - v0).norm(dim=1)
    C = (v0 - v1).norm(dim=1)

    # Area of each triangle (with Heron's formula); shape is (sum(F_n),)
    s = 0.5 * (A + B + C)
    # note that the area can be negative (close to 0) causing nans after sqrt()
    # we clip it to a small positive value
    # pyre-fixme[16]: `float` has no attribute `clamp_`.
    area = (s * (s - A) * (s - B) * (s - C)).clamp_(min=eps).sqrt()

    # Compute cotangents of angles, of shape (sum(F_n), 3)
    A2, B2, C2 = A * A, B * B, C * C
    cota = (B2 + C2 - A2) / area
    cotb = (A2 + C2 - B2) / area
    cotc = (A2 + B2 - C2) / area
    cot = torch.stack([cota, cotb, cotc], dim=1)
    cot /= 4.0

    # Construct a sparse adjacency matrix by basically doing:
    # A[v1, v2] = cota
    # A[v2, v0] = cotb
    # A[v0, v1] = cotc
    ii = face[0,:,[1, 2, 0]]
    jj = face[0,:,[2, 0, 1]]
    idx = torch.stack([ii, jj], dim=0).reshape(2, nf * 3)
    # pyre-fixme[16]: Module `sparse` has no attribute `FloatTensor`.
    A = torch.sparse.FloatTensor(
        idx, cot.reshape(-1), (nv, nv))
    # Make it symmetric: L[v1, v2] = L[v2, v1] = (cota + cota')/2
    A = 0.5*(A + A.t())   

    # construct degree matrix D
    degree = torch.sparse.sum(A, dim=-1).to_dense()
    # set the diagonal entries as the degree of vertices
    self_edge = torch.arange(nv)[None].repeat([2,1]).to(face.device)
    D = torch.sparse_coo_tensor(
        self_edge, degree, (nv, nv))
    L = (D - A).unsqueeze(0)
    
    # For each vertex, compute the sum of areas for triangles containing it.
    idx = face.reshape(-1)
    inv_area = torch.zeros(nv).float().to(vert.device)
    val = torch.stack([area] * 3, dim=1).reshape(-1)
    inv_area.scatter_add_(0, idx, val)
    idx = inv_area > 0
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    inv_area[idx] = 1.0 / inv_area[idx]
    inv_area = inv_area.reshape(1, -1, 1)

    return L, inv_area, degree.reshape(1,-1,1)


def laplacian_smooth(vert, face, lambd=1., n_iters=1):
    """
    Laplacian mesh smoothing.
    
    Inputs:
    - vert: input mesh vertices, (1,|V|,3) torch.Tensor
    - face: input mesh faces, (1,|F|,3) torch.LongTensor
    - lambd: strength of mesh smoothing [0,1]
    - n_iters: number of mesh smoothing iterations
    
    Returns:
    - vert: smoothed mesh vertices, (1,|V|,3) torch.Tensor
    """
    L = laplacian(face)
    for n in range(n_iters):
        vert = vert - lambd * L.bmm(vert)
    return vert


def cot_laplacian_smooth(vert, face, lambd=1., n_iters=1):
    """
    Cotangent Laplacian mesh smoothing.
    
    Inputs:
    - vert: input mesh vertices, (1,|V|,3) torch.Tensor
    - face: input mesh faces, (1,|F|,3) torch.LongTensor
    - lambd: strength of mesh smoothing [0,1]
    - n_iters: number of mesh smoothing iterations
    
    Returns:
    - vert: smoothed mesh vertices, (1,|V|,3) torch.Tensor
    """
    for n in range(n_iters):
        # the cot laplacian matrix need to be updated each iter
        with torch.no_grad():
            # compute cotangent laplaican matrix
            L, _, degree = cot_laplacian(vert, face)
        vert = vert - lambd * L.bmm(vert) / degree
    return vert


def taubin_smooth(vert, face, lambd=0.5, mu=-0.53, n_iters=1):
    """
    Taubin mesh smoothing.
    
    Inputs:
    - vert: input mesh vertices, (1,|V|,3) torch.Tensor
    - face: input mesh faces, (1,|F|,3) torch.LongTensor
    - lambd: strength of mesh smoothing [0,1]
    - mu: strength of mesh smoothing [-1,0]
    - n_iters: number of mesh smoothing iterations
    
    Returns:
    - vert: smoothed mesh vertices, (1,|V|,3) torch.Tensor
    """
    L = laplacian(face)
    for n in range(n_iters):
        vert = vert - mu * L.bmm(vert)
        vert = vert - lambd * L.bmm(vert)
    return vert


def area_weighted_smooth(vert, face, strength=1., n_iters=1):
    """
    Area weighted surface smoothing.
    
    This function reimplements the method in the connectome
    workbench commandline. For originial code please see: 
    https://github.com/Washington-University/workbench/blob/
    master/src/Algorithms/AlgorithmSurfaceSmoothing.cxx

    Inputs:
    - vert: input mesh vertices, (1,|V|,3) torch.Tensor
    - face: input mesh faces, (1,|F|,3) torch.LongTensor
    - strength: strength of mesh smoothing [0,1]
    - n_iters: number of mesh smoothing iterations
    
    Returns:
    - vert: smoothed mesh vertices, (1,|V|,3) torch.Tensor
    """
    
    nv = vert.shape[1]
    nf = face.shape[1]
    fid = torch.arange(nf).to(face.device)
    v_f_id = torch.cat([torch.stack([face[0,:,0], fid]),
                        torch.stack([face[0,:,1], fid]),
                        torch.stack([face[0,:,2], fid])], dim=-1)
    # adjacency matrix that maps faces to vertices:
    # it contains the neighborhood face indices for each vertex
    A = torch.sparse_coo_tensor(
        v_f_id, torch.ones_like(v_f_id[0]).float(), (nv, nf)).unsqueeze(0)

    for n in range(n_iters):
        v_f = vert[:, face[0]]  # vertices of each face (1,|F|,|V|,3)
        f_center = v_f.mean(-2)  # center of each face
        f_area = face_area(vert, face).reshape(1,-1,1)  # area of each face
        vert_avg = A.bmm((f_center * f_area)) / A.bmm(f_area)
        vert = (1-strength)*vert + strength*vert_avg
    return vert
