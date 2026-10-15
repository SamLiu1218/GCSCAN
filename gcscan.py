import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from multiprocessing import Pool
from itertools import repeat
import networkx as nx
from sklearn.cluster import SpectralClustering
from datetime import datetime
from scipy.spatial import Delaunay

def report(str):
    from datetime import datetime
    current_time = datetime.now().strftime("%H:%M:%S")
    print(current_time, "  ", str)

def axisoff():
    import matplotlib.pyplot as plt
    plt.tick_params(axis='both', which='both', top=False,
                    bottom=False, left=False, right=False,
                    labelbottom=False, labelleft=False)
    
def Neighbors_radius(xy, radius=30):
    # xy = adata.obsm['spatial']
    neigh = NearestNeighbors(radius=radius, algorithm='auto')
    neigh.fit(xy)
    
    # Get neighbors and distances
    distances, indices = neigh.radius_neighbors(xy, return_distance=True)
    
    # Build edge list with distances
    El = [
        [cell_idx, v, dist] 
        for cell_idx, (neighbors, dists) in enumerate(zip(indices, distances))
        for v, dist in zip(neighbors, dists) 
        if v != cell_idx
    ]
    
    # Create DataFrame
    E_df = pd.DataFrame(El, columns=['Cell_id_1', 'Cell_id_2', 'Distance'])
    E_df = E_df.astype({'Cell_id_1': 'int', 'Cell_id_2': 'int'})  # Ensure Cell_id_1 and Cell_id_2 are integers
    return E_df


def calculate_min_distance_cdist(coords1, coords2):
    dist = cdist(coords1, coords2)
    min_dist = np.min(dist)
    return min_dist

def process_chunk(df_chunk, coords):
    results = []
    for idx in df_chunk.index:
        coords_1, coords_2 = coords[df_chunk.loc[idx, ['Cell_id_1', 'Cell_id_2']].astype(int)]
        boundary_dist = calculate_min_distance_cdist(coords_1, coords_2)
        results.append((idx, boundary_dist))
    return results

def calculate_boundary_distance(df, coords, num_cpus):
    '''
    df: edge dataframe with columns ['Cell_id_1', 'Cell_id_2']
    coords: list of boundary coordinates
    '''
    df = df.query("Cell_id_2 > Cell_id_1").copy()

    chunk_size = int(np.ceil(len(df) / num_cpus))
    chunks = [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    
    with Pool(processes=num_cpus) as pool:
        # results = pool.map(process_chunk, chunks)
        results = pool.starmap(process_chunk, zip(chunks, repeat(coords)))
    
    results = [item for sublist in results for item in sublist]  # Flatten the list of lists
    for idx, boundary_dist in results:
        df.at[idx, 'boundary_dist'] = boundary_dist

    

    df_swapped = df.copy()
    df_swapped.columns = ['Cell_id_2', 'Cell_id_1', 'boundary_dist']
    df = pd.concat([df, df_swapped], ignore_index=True)
    return df
    
def Neighbors_delaunay(xy, radius = 25):
    # Calculate the Delaunay triangulation
    tri = Delaunay(xy)

    # Extract edges with filtering by radius
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            edge = (simplex[i], simplex[(i + 1) % 3])
            if edge not in edges and tuple(reversed(edge)) not in edges:
                point1 = xy[edge[0]]
                point2 = xy[edge[1]]
                distance = np.linalg.norm(point1 - point2)
                if distance <= radius:
                    edges.add((*edge, distance))

    # Convert edges to DataFrame
    El = list(edges)
    E_df = pd.DataFrame(El, columns=['Cell_id_1', 'Cell_id_2', 'Distance'])
    E_df = E_df.astype({'Cell_id_1': 'int', 'Cell_id_2': 'int'})
    return E_df


def process_simplex(simplex, xy, radius):
    edges = []
    for i in range(3):
        edge = (simplex[i], simplex[(i + 1) % 3])
        point1 = xy[edge[0]]
        point2 = xy[edge[1]]
        distance = np.linalg.norm(point1 - point2)
        if distance <= radius:
            # Ensure edge is sorted and include the distance
            edges.append((int(edge[0]), int(edge[1]), distance))
    return edges

def Neighbors_delaunay_parallel(xy, radius=30, num_cpus=4):
    # Calculate the Delaunay triangulation
    tri = Delaunay(xy)

    # Use multiprocessing Pool to process simplices in parallel
    with Pool(processes=num_cpus) as pool:
        results = pool.starmap(process_simplex, [(simplex, xy, radius) for simplex in tri.simplices])

    # Combine results and remove duplicates
    edges = set()
    for sublist in results:
        for edge in sublist:
            # # Add edge as a sorted tuple to ensure uniqueness
            edge_sorted = tuple(sorted(edge[:2])) + (edge[2],)
            edges.add((edge_sorted[0], edge_sorted[1], edge_sorted[2]))
            edges.add((edge_sorted[1], edge_sorted[0], edge_sorted[2]))
            # edges.add(tuple(edge))

    # Convert edges to DataFrame
    El = list(edges)
    E_df = pd.DataFrame(El, columns=['Cell_id_1', 'Cell_id_2', 'Distance'])
    E_df = E_df.astype({'Cell_id_1': 'int', 'Cell_id_2': 'int'})  # Ensure integer IDs
    return E_df
    
def calculate_edges(xy, coords = None, num_cpus = 1, radius = 25, method = 'boundary'):
    '''method = ['boundary', 'center', 'delaunay']'''
    if method not in ['boundary', 'center', 'delaunay']:
        raise ValueError("method must be 'boundary', 'center', or 'delaunay'")
    if method == 'delaunay':
        E_df = Neighbors_delaunay_parallel(xy, radius, num_cpus)
    else:
        E_df = Neighbors_radius(xy, radius)
        if method == 'boundary':
            E_df = calculate_boundary_distance(E_df, coords, num_cpus)
    return E_df


# Union-Find (Disjoint Set) implementation
class UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, item):
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, set1, set2):
        root1 = self.find(set1)
        root2 = self.find(set2)
        if root1 != root2:
            if self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
            elif self.rank[root1] < self.rank[root2]:
                self.parent[root1] = root2
            else:
                self.parent[root2] = root1
                self.rank[root1] += 1


def graph_watershed(G, high_value_nodes, weight_threshold=0.3):
    import heapq
    uf = UnionFind()
    itrt  = 0
    for node in G.nodes():
        uf.parent[node] = node
        uf.rank[node] = 0

    pq = []  # This is our priority queue

    # Initialize the sources in the priority queue with label equal to their node id
    for source in high_value_nodes:
        heapq.heappush(pq, (-1, source, source))  # (negative value for priority, current node, label)
    
    while pq:
        itrt += 1
        # if itrt < 100 and itrt%5 == 0:
        #     zut.report(f"{itrt}, {len(pq)}")
        curr_priority, curr_node, curr_label = heapq.heappop(pq)

        # If the current node has already been labeled by another source, skip it
        if uf.find(curr_node) != curr_label:
            continue

        # Process each neighbor of the current node
        for neighbor in G.neighbors(curr_node):
            edge_weight = G[curr_node][neighbor]['weight']

            # Only consider neighbors with a high enough edge weight
            if edge_weight < weight_threshold:
                continue

            if uf.find(neighbor) != curr_label:
                uf.union(curr_node, neighbor)
                heapq.heappush(pq, (curr_priority - edge_weight, neighbor, curr_label))

    # Create the final labels dictionary from the union-find structure
    labels = {node: uf.find(node) for node in G.nodes()}

    # zut.report(f"{itrt} iterations in total.")
    return labels


# Graph-based erosion
def graph_erosion(G, cluster_nodes):
    eroded_nodes = set(cluster_nodes)
    boundary_nodes = set()
    for node in cluster_nodes:
        if any(neighbor not in cluster_nodes for neighbor in G.neighbors(node)):
            boundary_nodes.add(node)
    eroded_nodes.difference_update(boundary_nodes)
    return list(eroded_nodes)

# Function to keep the largest connected component
def keep_largest_component(G, nodes):
    subgraph = G.subgraph(nodes)
    largest_component = max(nx.connected_components(subgraph), key=len)
    return list(largest_component)

# Graph-based dilation
def graph_dilation(G, cluster_nodes):
    dilated_nodes = set(cluster_nodes)
    for node in cluster_nodes:
        neighbors = set(G.neighbors(node))
        dilated_nodes.update(neighbors)
    return list(dilated_nodes)

def postprocessing(G, V_df, c):
    labels = {}
    for cluster_label in V_df[c].value_counts().index.values[1:]:
        nodes = V_df[V_df[c] == cluster_label].index.tolist()
        nodes = graph_erosion(G, nodes)
        try:
            nodes = keep_largest_component(G, nodes)
        except:
            pass
        nodes = graph_dilation(G, nodes)
        nodes = graph_dilation(G, nodes)
        nodes = graph_erosion(G, nodes)
        for i in nodes:
            labels[i] = cluster_label
    
    return V_df.index.map(labels).fillna(0).astype(int)


def calculate_cc_edge_ratio(G, labels, core_nodes):
    cut_edges = []
    cc_cut_edges = []

    for u, v, data in G.edges(data=True):
        if labels.loc[u] != labels.loc[v]:
            cut_edges.append((u, v))
            if u in core_nodes and v in core_nodes:
                cc_cut_edges.append((u, v))

    total_cut_edges = len(cut_edges)
    cc_cut_edges_count = len(cc_cut_edges)

    if total_cut_edges == 0:
        return 0  # Avoid division by zero

    cc_edge_ratio = cc_cut_edges_count / total_cut_edges
    return cc_edge_ratio, total_cut_edges, cc_cut_edges_count


def NCut(G, V_df, col_cluster, cluster_label, key_added = 'ncut_label'):
    cluster_nodes = V_df[V_df[col_cluster] == cluster_label].index.tolist()
    
    G_sub = G.subgraph(cluster_nodes).copy()
    
    # Adjacency matrix
    # adjacency_matrix = nx.to_numpy_array(G_sub, nodelist=cluster_nodes)
    adjacency_matrix = nx.to_scipy_sparse_array(G_sub, nodelist=cluster_nodes, format='csr')
    # adjacency_matrix = adjacency_matrix.astype('int32') 
    
    # Convert indices and indptr to int32, keeping data as float64
    adjacency_matrix.indices = adjacency_matrix.indices.astype('int32')
    adjacency_matrix.indptr = adjacency_matrix.indptr.astype('int32')

    # print(adjacency_matrix.data.dtype)  # Should be float64 or float32 (based on your graph's weights)
    # print(adjacency_matrix.indices.dtype)  # Should be int32
    # print(adjacency_matrix.indptr.dtype)  # Should be int32
        
    # Apply Spectral Clustering with precomputed affinity
    spectral = SpectralClustering(n_clusters=2, affinity='precomputed', assign_labels='discretize', random_state=42)
    labels = spectral.fit_predict(adjacency_matrix)

    
    # Create sets for the two partitions
    A = [cluster_nodes[i] for i in range(len(cluster_nodes)) if labels[i] == 0]
    B = [cluster_nodes[i] for i in range(len(cluster_nodes)) if labels[i] == 1]
    
    # Calculate cut(A, B)
    cut_value = 0
    for u, v, data in G_sub.edges(data=True):
        if (u in A and v in B) or (u in B and v in A):
            cut_value += data['weight']
    
    # Calculate assoc(A, V) and assoc(B, V)
    assoc_A = sum(degree for node, degree in nx.degree(G_sub, A, weight='weight'))
    assoc_B = sum(degree for node, degree in nx.degree(G_sub, B, weight='weight'))
    
    
    # Calculate Ncut
    ncut_value = (cut_value / assoc_A) + (cut_value / assoc_B)

    sub_vdf = V_df.loc[cluster_nodes].copy()
    sub_vdf[key_added] = labels

    core_nodes = sub_vdf.loc[sub_vdf['iscore']].index
    cc_edge_ratio, total_cut_edges, cc_cut_edges_count = calculate_cc_edge_ratio(G_sub, sub_vdf[key_added], core_nodes)
    return sub_vdf, ncut_value, cc_edge_ratio, total_cut_edges

def relabel_by_size(sr, bg = 0):
    label_ordered = [v for v in sr.value_counts().index.values if v != bg]
    mapping = {v:i + 1 for i, v in enumerate(label_ordered)}
    mapping[bg] = 0
    return sr.replace(mapping)


def gcscan(
    xy, 
    coords, 
    involved, 
    num_cpus = 1,
    radius = 25, 
    boundary_dist_threshold = 10, 
    boundary_dist_factor = -0.4,
    minimal_neighbor_count = 5,
    watershed_min_weight = 0.4, 
    watershed_initial_threshold = 0.6,
    ncut_threshold = 0.02, 
    iscore = None, 
    verbose = False,
    edge_method = 'boundary',
    max_cutting_size = 40000,
):
    if iscore is None:
        iscore = np.array([False]*len(involved))

    # Initiate E_df
    E_df = calculate_edges(
        xy = xy, 
        coords = coords,
        num_cpus = num_cpus,
        method = edge_method,
        radius = radius,
)
    
    if verbose:
        report(f"{len(xy)} cells, {len(E_df)} edges.")
        
    # Filter by boundary_dist
    if edge_method == 'boundary':
        E_df = E_df.query("boundary_dist <= @boundary_dist_threshold").copy()
        E_df['factor'] = np.exp(boundary_dist_factor * E_df['boundary_dist'])
        E_df.loc[E_df['factor'] <= 0, 'factor'] = 0
        
        if verbose:
            report(f"{len(E_df)} refiend edges within boundary dist threshold.")
            
    else:
        E_df['factor'] = np.exp(-(E_df['Distance']/radius)**2)

    
    # V_df 
    E_df['Cell_type_2'] = involved[E_df['Cell_id_2'].values]
    
    if len(np.unique(involved)) == 2:
        V_df = pd.DataFrame({
            'Neighbor_count': E_df['Cell_id_1'].value_counts(), 
            'involved_count': E_df.loc[E_df['Cell_type_2'], 'Cell_id_1'].value_counts(),
        }).fillna(0)
        V_df[['x','y']] = xy[V_df.index]
        V_df['involved'] = involved[V_df.index]
        V_df['iscore'] = iscore[V_df.index]
    
        
        V_df['Neighbor_count_adj'] = V_df['Neighbor_count'].copy()
        V_df.loc[V_df['Neighbor_count_adj'] <= minimal_neighbor_count,'Neighbor_count_adj'] = minimal_neighbor_count
        V_df['Involvement'] = (V_df['involved_count']/V_df['Neighbor_count_adj']).fillna(0)
    else:
        report("Treating involvement as numeric instead of bool values")
        
        V_df = pd.DataFrame({
            'Neighbor_count': E_df['Cell_id_1'].value_counts(), 
            'involved_count': E_df.groupby('Cell_id_1')['Cell_type_2'].sum(),
        }).fillna(0)
        
        V_df[['x','y']] = xy[V_df.index]
        V_df['involved'] = involved[V_df.index]
        V_df['iscore'] = iscore[V_df.index]
    
        
        V_df['Neighbor_count_adj'] = V_df['Neighbor_count'].copy()
        V_df.loc[V_df['Neighbor_count_adj'] <= minimal_neighbor_count,'Neighbor_count_adj'] = minimal_neighbor_count
        V_df['Involvement'] = (V_df['involved_count']/V_df['Neighbor_count_adj']).fillna(0)
    
    E_df['weight'] = np.sqrt(V_df.loc[E_df['Cell_id_1'], 'Involvement'].values * V_df.loc[E_df['Cell_id_2'], 'Involvement'].values)
    E_df['weight'] = E_df['weight']*E_df['factor']
    V_df['weights_max'] = E_df.groupby('Cell_id_1')['weight'].max()
    

    # Graph construction

    G = nx.from_pandas_edgelist(E_df, 'Cell_id_1', 'Cell_id_2', ['weight'])
    
    # Add node attributes
    for cell_id, row in V_df.iterrows():
        G.add_node(cell_id, 
                   Involvement = row['Involvement'],
                   involved = row['involved'],
                   x = row['x'],
                   y = row['y'])
    


    if verbose:
        report(f"graph constructed. Running global clustering")
    
    
    high_value_nodes = V_df.loc[V_df['weights_max'] >= watershed_initial_threshold].index.tolist()

    high_value_nodes = V_df.loc[V_df['Involvement'] >= watershed_initial_threshold].index.tolist()
    

    
    # Run the watershed algorithm
    labels = graph_watershed(G, high_value_nodes, watershed_min_weight)

    c = 'cluster_watershed'
    V_df[c] = V_df.index.map(labels).fillna(0).astype(int)
    V_df[c] = V_df[c].replace({-1:0})
    V_df.loc[V_df[c].isin(V_df[c].value_counts().loc[V_df[c].value_counts() <= 10].index), c] = 0
    V_df[c] = V_df[c].replace({v:i for i, v in enumerate(V_df[c].value_counts().index)})

    
    if verbose:
        report(f"{V_df[c].max() - 1} clusters detected. Partitioning...")

    V_df['cluster_watershed_ncut'] = V_df['cluster_watershed'].copy()

    ## NCUT!

    cut_df = pd.DataFrame(columns = ['parent','children1','children2','ncut','cc_edge_ratio', 'cut'])
    
    c = 'cluster_watershed_ncut'
    clusters_todo = set(V_df.loc[V_df[c] > 0, c].unique())

    while len(clusters_todo) > 0:
        cut_df_idx = len(cut_df)
        
        cluster_label = list(clusters_todo)[0]
        clusters_todo.remove(cluster_label)
        if np.sum(V_df[c] == cluster_label) < 100:
            continue
        
        if np.sum(V_df[c] == cluster_label) > max_cutting_size:
            # too large
            continue
        sub_vdf, ncut_value, cc_edge_ratio, total_cut_edges = NCut(G, V_df, col_cluster = c, cluster_label = cluster_label)
        V_df['ncut_label'] = sub_vdf['ncut_label'] + 1

        cut_df.loc[cut_df_idx, ['parent','ncut','cc_edge_ratio', 'cut']] = [cluster_label, ncut_value, cc_edge_ratio, False]
        if (ncut_value < ncut_threshold) and (cc_edge_ratio < 0.02) and (sub_vdf['ncut_label'].value_counts().min() > 50):
            cut_df.loc[cut_df_idx, ['children1','children2', 'cut']] = [1 + V_df[c].max(), 2 + V_df[c].max(),  True]
            clusters_todo.add(1 + V_df[c].max())
            clusters_todo.add(2 + V_df[c].max())
            V_df.loc[sub_vdf.index, c] = sub_vdf['ncut_label'] + 1 + V_df[c].max()


    
    if verbose:
        report(f"{(relabel_by_size(V_df[c])).max() - 1} clusters partitioned.")
    # Post processing
    
    V_df['cluster_watershed_ncut_post'] = postprocessing(G, V_df, 'cluster_watershed_ncut')
    c = 'cluster_watershed_ncut_post'
    
    V_df['cluster_watershed_ncut_post_50'] = 0
    pass_index = V_df[c].value_counts().loc[V_df[c].value_counts()>50].index
    V_df.loc[V_df[c].isin(pass_index), 'cluster_watershed_ncut_post_50'] = V_df.loc[V_df[c].isin(pass_index), c]
    
    V_df['cluster_watershed_ncut_post_50'] = relabel_by_size(V_df['cluster_watershed_ncut_post_50'])

    if verbose:
        report(f"{V_df['cluster_watershed_ncut_post_50'].max() - 1} final clusters.")

    V_df['cluster_final'] = V_df['cluster_watershed_ncut_post_50']
    
    return E_df, V_df, cut_df

    


def alpha_shape_with_area(points, alpha=None):
    import numpy as np
    from scipy.spatial import Delaunay
    from sklearn.neighbors import NearestNeighbors


    def add_edge(edges, edge_points, i, j, points):
        if (i, j) in edges or (j, i) in edges:
            # already added
            edges.remove((j, i) if (j, i) in edges else (i, j))
            return
        edges.add((i, j))
        edge_points.append(points[[i, j]])

    points = np.array(points)
    if alpha is None:
        nbrs = NearestNeighbors(n_neighbors=2).fit(points)
        distances, _ = nbrs.kneighbors(points)
        avg_distance = np.mean(distances[:, 1])  
        alpha = 1 / avg_distance

    n_try = 0
    try:
        tri = Delaunay(points)
        edges = set()
        edge_points = []
        area = 0.0
    
        for ia, ib, ic in tri.simplices:
            pa = points[ia]
            pb = points[ib]
            pc = points[ic]
            
            a = np.linalg.norm(pa - pb)
            b = np.linalg.norm(pb - pc)
            c = np.linalg.norm(pc - pa)
            s = (a + b + c) / 2.0
            triangle_area = max(s * (s - a) * (s - b) * (s - c), 0) 
    
            circum_r = a * b * c / (4.0 * np.sqrt(triangle_area)) if triangle_area > 0 else np.inf
            if circum_r < 1.0 / alpha:
                add_edge(edges, edge_points, ia, ib, points)
                add_edge(edges, edge_points, ib, ic, points)
                add_edge(edges, edge_points, ic, ia, points)
                area += triangle_area
    
        
        boundary_edges = np.vstack(edge_points)
        unique_boundary_points = np.unique(boundary_edges.reshape(-1, 2), axis=0)
    except:
        n_try += 1
        if n_try >= 10:
            raise ValueError(f'alpha = {alpha} too small')
        else:
            alpha = alpha/2
            unique_boundary_points, area = alpha_shape_with_area(points, alpha=alpha)

    return unique_boundary_points, area


def calculate_convex_hull_area(points):
    from scipy.spatial import ConvexHull
    '''coords, area = calculate_convex_hull_area(xy)'''
    hull = ConvexHull(points)
    coords = hull.points[hull.vertices]
    coords = np.vstack([coords, coords[0]])
    area = hull.volume
    return coords, area

def calculate_alpha_convex_hull_area(points, alpha = None):
    '''coords, area = calculate_alpha_convex_hull_area(xy)'''
    boundary, _ = alpha_shape_with_area(points, alpha = alpha)
    return calculate_convex_hull_area(boundary)

def isinROI(xy, roi_coords):
    from matplotlib.path import Path
    roi_path = Path(roi_coords)
    inside = roi_path.contains_points(xy)
    return inside

def calculate_GC_df(V_df, cdf, col = 'cluster_final'):
    
    GC_df = V_df.loc[V_df[col] > 0, col].value_counts().to_frame('n_cell')
    GC_df[['Area', 'GC_area', ]] = 0
    coords_ls = []
    
    for gc_id in GC_df.index:
        cdf_cur = cdf.reset_index().loc[V_df.index].loc[V_df[col] == gc_id]
        coords, area = calculate_alpha_convex_hull_area(cdf_cur[['x','y']].values)
        gc_ncell = np.sum(V_df.loc[cdf_cur.index, 'iscore'])
        
        if gc_ncell > 2:
            gc_coords, gc_area = calculate_alpha_convex_hull_area(V_df.loc[cdf_cur.index].query('iscore')[['x','y']].values)
            n_GC_inGC = np.sum(isinROI(V_df.query("iscore")[['x','y']].values, gc_coords))
            n_any_inGC = np.sum(isinROI(V_df[['x','y']].values, gc_coords))
            GC_GC_density = n_GC_inGC/n_any_inGC
            if n_GC_inGC < 10 or GC_GC_density < 0.1: # too small or sparse
                gc_area = 0
            
        else:
            gc_area = 0
        GC_df.loc[gc_id, ['Area', 'GC_area', 'GC_ncell' ]] = area, gc_area, gc_ncell
        coords_ls.append(coords)
    GC_df['coords'] = coords_ls
    GC_df['GC_ratio'] = GC_df['GC_area']/GC_df['Area']
    return GC_df

def assignROIs(xy, roi_coords_list, roi_names=None):
    '''
    Assign each point to an ROI from a list of ROIs.
    
    input:
        xy: array of shape (n_points, 2) with x, y coordinates.
        roi_coords_list: list of ROI coordinate arrays, each passed to isinROI.
        roi_names: optional list of names for each ROI. If None, uses 0-based integers.
    
    return: 1D array of length n_points with ROI identity. None if not in any ROI.
    
    e.g.:
        assignROI(xy, [roi1_coords, roi2_coords], roi_names=['tumor', 'stroma'])
    '''
    import warnings
    import numpy as np

    if roi_names is None:
        roi_names = list(range(len(roi_coords_list)))

    identity = np.full(len(xy), None, dtype=object)
    n_assigned = np.zeros(len(xy), dtype=int)

    for roi_name, roi_coords in zip(roi_names, roi_coords_list):
        inside = isinROI(xy, roi_coords)
        n_assigned += inside.astype(int)
        identity[inside] = roi_name

    n_overlap = (n_assigned > 1).sum()
    if n_overlap > 0:
        warnings.warn(f"{n_overlap} point(s) belong to more than one ROI and were assigned to the last matching ROI.")

    return identity
    
def assign_TLS_id(GC_df, cdf):
    clusters = assignROIs(
        xy = cdf[['x','y']].values,
        roi_coords_list = GC_df['coords'].values,
        roi_names = GC_df.index
    )
    return clusters