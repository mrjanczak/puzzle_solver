import os
import sys
import numpy as np
import cv2 as cv
import pickle
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from scipy.spatial.distance import euclidean
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from fastdtw import fastdtw

local = os.path.dirname(__file__)
# puzzle size
M, N = 40, 25

def get_corners(harris_corners, neighborhood_size=5, score_threshold=0.3, minmax_threshold=100):
    
    """
    Given the input Harris image (where in each pixel the Harris function is computed),
    extract discrete corners
    """
    data = harris_corners.copy()
    data[data < score_threshold*harris_corners.max()] = 0.

    data_max = filters.maximum_filter(data, neighborhood_size)
    maxima = (data == data_max)
    data_min = filters.minimum_filter(data, neighborhood_size)
    diff = ((data_max - data_min) > minmax_threshold)
    maxima[diff == 0] = 0

    labeled, num_objects = ndimage.label(maxima)
    yx = np.array(ndimage.center_of_mass(data, labeled, range(1, num_objects+1)))

    return yx[:, ::-1]

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    v_len = np.linalg.norm(vector)
    if abs(v_len) > 1e-3:
        return vector / v_len
    else:
        return vector

def angle(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'"""
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    # acos = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    # asin = np.arcsin(np.clip(np.cross(v1_u, v2_u), -1.0, 1.0)) 

    acos = np.arccos(np.dot(v1_u, v2_u))
    asin = np.arcsin(np.cross(v1_u, v2_u))
    return acos*np.sign(asin)*180/np.pi

def get_pieces(dir = 'img', pieces = [], plot_piece = True,
               gauss_blur = (5,5), 
               eps_ratio = .0004,
               num_points = 120,
               harris_kwargs = {'blockSize': 3, 'ksize': 5, 'k': .04},
               corner_peak_kwargs = {'height': 30, 'distance': 5}, 
               savgol_kwargs = {'window_length': 7, 'polyorder': 2},
               lock_peak_kwargs = {'height':60, 'distance':5},
               ):

    local = os.path.dirname(__file__)
    path = os.path.join(local, dir)
    os.chdir(local)

    filenames = os.listdir(path)
    filenames.sort()

    for piece_id, file in enumerate(filenames[1:2], start = len(pieces)):
        file = os.path.join(dir,file)
        print(file)
        img = cv.imread(file)

        # Find main piece
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray,gauss_blur,0)
        _,thresh = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)    
        thresh = cv.bitwise_not(thresh)
        ret, labels = cv.connectedComponents(thresh)
        connected_areas = [np.count_nonzero(labels == l) for l in range(1, ret)]
        max_area_idx = np.argmax(connected_areas) + 1
        gray[labels != max_area_idx] = 0
        gray[labels == max_area_idx] = 255
        
        # Find simplified contour
        _,thresh = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)    
        thresh = cv.bitwise_not(thresh)    
        cnts, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        largest_cnt = max(cnts, key=cv.contourArea)        
        epsilon = eps_ratio * cv.arcLength(largest_cnt, True)
        appx_cnt = cv.approxPolyDP(largest_cnt, epsilon, True)

        # Evenly distribute cnt
        arc_length = cv.arcLength(appx_cnt, True)
        step_size = arc_length / num_points  # Distance between points

        # Convert contour to NumPy format and interpolate points
        appx_cnt = appx_cnt[:, 0, :]  # Remove unnecessary dimensions
        curve = np.vstack([appx_cnt, appx_cnt[0]])  # Close the contour

        # Compute cumulative distances along the curve
        distances = np.cumsum(np.sqrt(np.sum(np.diff(curve, axis=0)**2, axis=1)))
        distances = np.insert(distances, 0, 0)  # Insert starting point

        # Interpolate points at evenly spaced distances
        new_points = np.zeros((num_points, 2), dtype=np.float32)
        for i, d in enumerate(np.linspace(0, distances[-1], num_points)):
            idx = np.searchsorted(distances, d)
            t = (d - distances[idx-1]) / (distances[idx] - distances[idx-1]) if idx > 0 else 0
            new_points[i] = (1 - t) * curve[idx-1] + t * curve[idx]

        # Convert back to contour format
        appx_cnt = new_points.astype(np.int32).reshape(-1, 1, 2)
        (cnt_x, cnt_y, cnt_w, cnt_h) = cv.boundingRect(appx_cnt)

        # Find corners
        blur = cv.GaussianBlur(gray, gauss_blur, 0)
        harris_corners = cv.cornerHarris(np.float32(blur), **harris_kwargs) 
        harris_corners = cv.dilate(harris_corners, None)
        threshold = 0.01 * harris_corners.max()
        corners = []
        for i in range(harris_corners.shape[0]):
            for j in range(harris_corners.shape[1]):
                if harris_corners[i, j] > threshold:                
                    corners.append([j, -i])
        corners = np.array(corners)
        corners = np.unique(corners, axis=0)

        # Prepare preview
        preview = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
        for P in corners:
            cv.circle(preview, (P[0], -P[1]), 5, (255, 0, 0), -1)
        cv.drawContours(preview, [appx_cnt], -1, (0, 255, 0), 3)
        # text = str(len(appx_cnt))
        # cv.putText(preview, text, (cnt_x+int(cnt_w/2), cnt_y+int(cnt_h/2)), cv.FONT_HERSHEY_SIMPLEX,
        #     0.9, (0, 255, 0), 2)

        # Unroll contour
        X = [v[0][0] for v in appx_cnt]
        Y = [-(v[0][1]) for v in appx_cnt]    
        V = [np.array([X[0]-X[-1], Y[0]-Y[-1]])]
        for x0,y0, x1,y1 in zip(X[:-1],Y[:-1],X[1:],Y[1:]):
            V.append([x1-x0, y1-y0])
        V.append(V[0])
        V = np.array(V)
        s_cum, a_cum = 0, 0
        s, a, a_ = [], [], []
        for id, (V1, V2) in enumerate(zip(V[:-1], V[1:])):
            s_i = np.linalg.norm(V1)
            a_i = angle(V1, V2)
            s_cum += s_i
            a_cum += a_i
            s.append(s_cum)
            a.append(a_i)
            a_.append(a_cum)

        # Find peaks of doubled a (true corners) 
        peaks_ids_, _ = find_peaks(a + a, **corner_peak_kwargs)
        peaks_ids = []
        ps, pa = [], []
        for id in peaks_ids_:
            if id >= len(s):
                id -= len(s)
            peaks_ids.append(id)
            ps.append(s[id])
            pa.append(a[id]) 

        # Find points of contour closest to corners
        cnt = []
        for x,y in zip(X, Y):
            cnt.append([x,y])
        cnt.append(cnt[0])
        cnt = np.array(cnt)

        cnt_cor = []
        cor_ids = []
        for P in corners:
            distances = np.linalg.norm(cnt - P, axis=1)
            id = np.argmin(distances)
            cnt_cor.append(cnt[id])
            cor_ids.append(id)
        cor_ids = list(set(cor_ids))
        cnt_cor = np.array(cnt_cor)

        # Remove false corners not in peaks (in reverse order)         
        for id in cor_ids[:]:
            if id not in peaks_ids:
                cor_ids.remove(id)          

        # Add corner points
        cs, ca, ca_ = [], [], []
        for id in cor_ids:
            cs.append(s[id])
            ca.append(a[id])
            ca_.append(a_[id])   

        # Extract edges
        edges = []
        flat_edges = []
        peaks_num = min(len(peaks_ids_), 4)
        for edge_id, (i0, i1) in enumerate(zip(peaks_ids_[0:peaks_num], peaks_ids_[1:peaks_num+1])):
            X__ = X+X
            Y__ = Y+Y
            eXY = np.column_stack((X__[i0:i1], Y__[i0:i1]))

            s__ = s+s
            a__ = a_+a_
            es = np.array(s__)[i0:i1] 
            ea = np.array(a__)[i0:i1] 
            es -= np.average(es)
            ea -= np.average(ea)

            ea_smooth = savgol_filter(ea, **savgol_kwargs) 
            eSA = np.column_stack((es, ea_smooth))

            # Identify lock by neg/pos peaks
            lock = 0
            _, neg_prop = find_peaks(-ea, **lock_peak_kwargs) 
            _, pos_prop = find_peaks(+ea, **lock_peak_kwargs)
            try:
                neg_peak_id = np.argmax(neg_prop['peak_heights'])
                pos_peak_id = np.argmax(pos_prop['peak_heights'])
            except:
                neg_peak_id, pos_peak_id = None, None
            n_es, n_ea = [], []
            p_es, p_ea = [], []

            if neg_peak_id and pos_peak_id:           
                n_es=[es[neg_peak_id]]
                n_ea=[ea[neg_peak_id]]
                p_es=[es[pos_peak_id]]
                p_ea=[ea[pos_peak_id]]

                # Set lock
                if neg_peak_id < pos_peak_id:
                    lock = +1
                else:
                    lock = -1

            if lock == 0:
                flat_edges.append(edge_id)

            edges.append({'eXY': eXY, 'eSA': eSA, 's': es , 'a_': ea, 'ea_smooth': ea_smooth, 'lock': lock,   'n_es': n_es, 'n_ea': n_ea, 'p_es': p_es, 'p_ea': p_ea })

        if len(flat_edges):
            edges__ = edges + edges
            id = max(flat_edges)+1
            edges = edges__[id: id+4]

        piece = {'id': piece_id, 'file': file, 'edges': edges, 'flat_edges': flat_edges}
        pieces.append(piece)

        if plot_piece:
            # Plot piece
            fig, axs = plt.subplots(2, 4, figsize=(10, 5))

            # Plot original points (with duplicates)
            axs[0,0].imshow(preview[cnt_y:cnt_y+cnt_h, cnt_x:cnt_x+cnt_w])
            axs[0,1].plot(cnt[:, 0], cnt[:, 1], color="blue", label="cnt", marker=".")
            axs[0,1].scatter(cnt_cor[:, 0], cnt_cor[:, 1], color="red",  label="corner", marker="X", s=20)

            # Plot unique points (after removing duplicates)
            axs[0,2].plot(s, a, color="blue")
            axs[0,2].scatter(ps, pa, color="green", marker="o")
            axs[0,2].scatter(cs, ca, color="red", marker="X", s=20)

            axs[0,3].plot(s, a_, color="blue")
            axs[0,3].scatter(cs, ca_, color="red", marker="X", s=20)

            for i, edge in enumerate(edges):
                color='gray'
                if edge['lock'] == 1:
                    color = 'red'
                elif edge['lock'] == -1:
                    color = 'blue'
                axs[1,i].plot(edge['s'], edge['a_'], color='gray')
                axs[1,i].plot(edge['s'], edge['ea_smooth'], color=color)
                axs[1,i].scatter(edge['n_es'], edge['n_ea'], color=color, marker=".")
                axs[1,i].scatter(edge['p_es'], edge['p_ea'], color=color, marker="+")
            
            # Show the plots
            plt.show()
    
    with open("pieces.pkl", "wb") as f:
        pickle.dump(pieces, f)

    return pieces

def init_puzzle(rows = M, cols = N, p_file = '', p_id = 0, p_row = 0, p_col = 0):

    puzzle = np.empty((rows, cols, 2), dtype=int)

    with open("pieces.pkl", "rb") as f:
        pieces = pickle.load(f)
    
    if not p_id and p_file:
        for piece in pieces:
            if piece['file'] == p_file:
                p_id = piece['id']
    # Set initial edge of 1st corner                
    e_id = 0

    # Write puzzle
    puzzle[p_row, p_col, 0] = p_id
    puzzle[p_row, p_col, 1] = e_id

    with open("puzzle.pkl", "wb") as f:
        pickle.dump(puzzle, f)

def match_frame():
    match_puzzle(col1 = 0, match_dir = 'TB:LR')
    match_puzzle(row0 = M, match_dir = 'TB:LR')
    match_puzzle(col0 = N, match_dir = 'BT:LR')
    match_puzzle(row1 = 0, match_dir = 'BT:LR')

def match_puzzle(row0 = 0, row1 = None, col0 = 0, col1 = None, match_dir = 'TB:LR', dist_crit = 1e3, confirm = True):
    """
    Default orientation of corner piece
        3
        -----
        |     |
    0  >     | 2
        |     |
        --^--
        1

    Default orientation of border piece
        3
        -----
        |     |
    0  >     < 2
        |     |
        --^--
        1      
    """
    # Read puzzle
    with open("puzzle.pkl", "rb") as f:
        puzzle = pickle.load(f)


    if match_dir == 'TB:LR' or 'BT:RL':
        dedge = +1
    else:
        dedge = -1

    dir = match_dir.split(':')

    with open("pieces.pkl", "rb") as f:
        pieces = pickle.load(f)

    if row1 is None:
        row1 = puzzle.shape[0]

    if col1 is None:
        col1 = puzzle.shape[1]


    if dir[0] == 'TB':
        row00 = row0
        rows = range(row0, row1+1)
        p1_off = (-1, 0)
    else:
        row00 = row1
        rows = range(row1, row0-1, -1)
        p1_off = (+1, 0)

    if dir[1] == 'LR':
        col00 = col0
        cols = range(col0, col1)
        p2_off = (0,-1)
    else:
        col00 = col1
        cols = range(col1, col0-1, -1)
        p2_off = (0,+1)
       

    for col in cols:
        for row in rows:
            
            # Skip initial piece
            if col == col00 and row == row00:
                continue
            
            # Get upper/lower reference piece - p1
            r,c = row + p1_off[0], col + p1_off[1]
            p1_id = puzzle[r, c, 0]

            if not p1_id:
                raise ValueError(f'Empty reference piece (row {r}, col {c})')
            
            e1_id = puzzle[r, c, 1]

            # Get opposite edge to matched previously in top-bottom or bottom-top match
            if not (r == row00 and c == col00):
                e1_id = e1_id+2 if e1_id+2 < 4 else e1_id-2

            edge = pieces[p1_id]['edges'][e1_id]
            lock1 = edge['lock']
            eSA1 = edge['eSA']

            # Skip internal pieces if frame evaluated
            if (col == 0 or col == puzzle.shape[1]-1) and len(piece['flat_edges']) == 0:
                continue
            
            # Get left/right reference piece - p2
            p2_id, e2_id = None, None
            lock2, eSA2 = None, None
            r,c = row + p2_off[0], col + p2_off[1]
            if (col > col00 and dir[1] == 'LR') or (col < col00 and dir[1] == 'RL'):
                p2_id = puzzle[r, c, 0]
                
                if p2_id:
                    # Get opposite edge to matched previously in left-right or right-left match
                    e2_id = puzzle[r, c, 1]
                    if dir[1] == 'LR':
                        e2_id = e2_id+1 if e2_id+1 < 4 else e2_id-3
                    else:
                        e2_id = e2_id+3 if e2_id+3 < 4 else e2_id-1

                    edge = pieces[p2_id]['edges'][e2_id]
                    lock2 = edge['lock']
                    eSA2 = edge['eSA']

            # Check all available pieces
            candidates = []
            for piece in pieces:
                p_id = piece['id']
                
                # Skip pieces already used
                if p_id in puzzle[:,:,0]:
                    continue
                
                # Try to match all edges with lock1/lock2
                for e_id in range(4):

                    # Get first edge to match with p2/e2
                    edge = pieces[p_id]['edges'][e_id]
                    lock = edge['lock']
                    eSA = edge['eSA']
                    eSA_inv = np.column_stack((eSA[::-1,0], -eSA[::-1,1]))
                    dist1, _ = fastdtw(eSA1, eSA_inv, dist=euclidean)

                    # Skip flat or not-matching edges
                    if lock == 0 or lock == lock1:
                        continue
                    
                    # Get adjacent edge to match with p2/e2
                    if lock2:
                        e_adj = e_id+dedge
                        if e_adj > 3:
                            e_adj -= 4
                        edge_adj = pieces[p_id]['edges'][e_adj]
                        lock_adj = edge['lock']                    

                        # Skip flat or not-matching edges
                        if lock_adj == 0 or lock_adj == lock2:
                            continue
 
                        eSA = edge_adj['eSA']
                        eSA_inv = np.column_stack((eSA[::-1,0], -eSA[::-1,1]))                        
                        dist2, _ = fastdtw(eSA2, eSA_inv, dist=euclidean)
                    else:
                        dist2 = 0
                    
                    # Measure how firmly pieces match to each other
                    dist = (dist1**2 + dist2**2)**.5

                    # Append potential candidate
                    if dist < dist_crit:
                        candidate = {'p_id': p_id, 'e_id': e_id, 'dist': dist}
                        candidates.append(candidate)
            
            # Sort candidates
            candidates = sorted(candidates, key=lambda c: c["dist"])
            
            # Plot candidates
            max_cand = min(4, len(candidates))
            fig, axs = plt.subplots(1, max_cand, figsize=(5, 5))
            for c, candidate in enumerate(candidates):
                print(candidate)

                p_id = candidate['p_id']
                e_id = candidate['e_id']
                eXY = pieces[p_id]['edges'][e_id]['eXY']

                for e_id_ in range(4):
                    color='gray'
                    if e_id == e_id_:
                        color = 'green'
                    axs[1,c].plot(eXY[:,0], eXY[:,1], color=color)
                axs[1,c].set_title(f'{p_id}:{e_id}')
            plt.show()

            c_id = 0
            p_id = candidates[c_id]['p_id']
            e_id = candidates[c_id]['e_id']

            # Ask user to confirm final choice of candidate
            if confirm:
                c_id = int(input(f'Input candidate id (0 - {p_id}:{e_id})', ).strip() or -1)
                if 0 >= c_id < len(candidates):
                    p1_id = candidates[c_id]['p_id']
                    e1_id = candidates[c_id]['e_id'] 
                else:
                    raise IndexError(f'Candidate id out of range')

            # Save matched piece
            puzzle[row, col, 0] = p_id
            puzzle[row, col, 1] = e_id

    with open("puzzle.pkl", "wb") as f:
        pickle.dump(puzzle, f)

    print(puzzle[:,:,1])

    return puzzle      


pieces = get_pieces(dir = 'img')
# pieces = get_pieces(dir = 'frame')
# pieces = get_pieces(dir = 'internal', pieces = pieces)

# init_puzzle(p_file = 'IMG_20250209_094955') 
# match_frame()
# match_puzzle(row0 = 1, row1 = M - 1, col0 = 1, col1 = N - 1)
