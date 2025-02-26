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

def get_pieces( dir = 'img', pieces = [], plot_piece = True,
                gauss_blur = (5,5), 
                eps_ratio = .0004,
                num_points = 160,
                corner_peak_kwargs = { 'distance': 5}, 
                savgol_kwargs = {'window_length': 7, 'polyorder': 2},
                lock_peak_kwargs = {'height':60, 'distance':5},

                lock_n = 10,
                lock_angle = 200,
                lock_max_mse = 200,
                lock_min_s = 1/16,   

                line_n = 2,
                line_max_mse = 10,
               ):

    local = os.path.dirname(__file__)
    path = os.path.join(local, dir)
    os.chdir(local)

    filenames = os.listdir(path)
    filenames.sort()

    for piece_id, file in enumerate(filenames[:1], start = len(pieces)):
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

        # Convert contour to NumPy format and compute cumulative distances along the curve
        appx_cnt = np.reshape(appx_cnt, (-1, 2))  # Remove unnecessary dimensions
        curve = np.vstack([appx_cnt, appx_cnt[0]])  # Close the contour
        s = np.cumsum(np.sqrt(np.sum(np.diff(curve, axis=0)**2, axis=1)))
        s = np.insert(s, 0, 0)  # Insert starting point

        # Interpolate points at evenly spaced distances
        int_cnt = np.zeros((num_points+1, 2), dtype=np.float32)
        for i, d in enumerate(np.linspace(0, s[-1], num_points+1)):
            idx = np.searchsorted(s, d)
            t = (d - s[idx-1]) / (s[idx] - s[idx-1]) if idx > 0 else 0
            int_cnt[i] = (1 - t) * curve[idx-1] + t * curve[idx]

        # Convert back to contour format
        cnt = int_cnt[:num_points].astype(np.int32)
        (cnt_x, cnt_y, cnt_w, cnt_h) = cv.boundingRect(cnt.reshape(-1, 1, 2))
        # Prepare preview
        preview = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
        cv.drawContours(preview, [cnt.reshape(-1, 1, 2)], -1, (0, 255, 0), 3)

        # Initialize counts
        # line_points = 0
        # non_line_points = 0

        # Compute cumulative distances
        cnt2 = np.vstack([cnt, cnt])  
        s2 = np.cumsum(np.sqrt(np.sum(np.diff(cnt2, axis=0)**2, axis=1)))
        s2 = np.insert(s2, 0, 0)  # Insert starting point

        # Compute angle and line/non-line points ratio
        a2 = np.zeros(num_points*2)
        for i in range(1, num_points*2-1):  # Avoid first & last points
            p1 = cnt2[i - 1]  # Previous point
            p2 = cnt2[i]      # Current point
            p3 = cnt2[i + 1]  # Next point

            # Compute vectors
            v1 = p2 - p1
            v2 = p3 - p2

            # Compute cosine of angle using dot product
            dot_product = np.dot(v1, v2)
            cross_product = np.cross(v1, v2)
            mag_v1 = np.linalg.norm(v1)
            mag_v2 = np.linalg.norm(v2)

            if mag_v1 == 0 or mag_v2 == 0:  # Avoid division by zero
                angle_degrees = 0.0
            else:
                cos_angle = dot_product / (mag_v1 * mag_v2)
                angle = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * np.sign(cross_product)  # Convert to radians
                angle_degrees = np.degrees(angle)
            a2[i] = angle_degrees           

        # Find locks
        cum_a2 = np.cumsum(a2, axis = 0)
        lock_cnt2 = []
        lock_cnt2_ = []
        lock_sca2 = []
        lock_sca2_ = []
        locks = []
        lock_mse2 = np.zeros(num_points*2)
        
        i = lock_n - 1
        for _ in range(lock_n, num_points*2 - lock_n):
            i += 1
            lock_s =         s2[i - lock_n : i + lock_n]
            lock_cum_a = cum_a2[i - lock_n : i + lock_n]
            id_max = np.argmax(lock_cum_a)
            id_min = np.argmin(lock_cum_a)
            i0 = min(id_min, id_max) 
            i1 = max(id_min, id_max)
            d_cum_a = lock_cum_a[i1] - lock_cum_a[i0]
            d_s = lock_s[i1] - lock_s[i0]
            lock_dir = np.sign(d_cum_a)

            if abs(d_cum_a) > lock_angle and abs(d_s) > s[-1]*lock_min_s:
                
                # Limit to peak-peak range
                lock_s_ =         lock_s[i0:i1]
                lock_cum_a_ = lock_cum_a[i0:i1]
                a, b = np.polyfit(lock_s_, lock_cum_a_, 1)  # 1 means linear fit
                lock_line = a * lock_s_ + b
                mse = np.mean((lock_cum_a_ - lock_line) ** 2)   

                if mse < lock_max_mse:
            
                    # Extend lock range
                    for j0 in range(i0-1, i0 - lock_n, -1):
                        j = i - lock_n + j0
                        if cum_a2[j]*lock_dir < cum_a2[j+1]*lock_dir:
                            i0 = j0
                        else:
                            break

                    for j1 in range(i1, i1 + lock_n, 1):   
                        j = i - lock_n + j1
                        if cum_a2[j-1]*lock_dir < cum_a2[j]*lock_dir:
                            i1 = j1
                        else:
                            break               
                    
                    # Find middle point of lock
                    i0_ = i - lock_n + i0
                    i1_ = i - lock_n + i1
                    i =   int((i0_+i1_)/2)

                    lock_cnt2.append(cnt2[i])
                    lock_cnt2_.append(cnt2[i0_])
                    lock_cnt2_.append(cnt2[i1_])
                    lock_sca2.append([s2[i], cum_a2[i]])
                    lock_sca2_.append([s2[i0_], cum_a2[i0_]])
                    lock_sca2_.append([s2[i1_], cum_a2[i1_]])
                    locks.append({'i0':i0_, 'i': i, 'i1':i1_, 'dir': lock_dir})
                    lock_mse2[i] = mse                    

                # Jump to next lock
                i += lock_n

            # End loop
            if i >= num_points*2 - lock_n:
                break

        lock_cnt2 =  np.array(lock_cnt2).reshape(-1, 2)
        lock_cnt2_ = np.array(lock_cnt2_).reshape(-1, 2)
        lock_sca2 =  np.array(lock_sca2).reshape(-1, 2)
        lock_sca2_ = np.array(lock_sca2_).reshape(-1, 2)
       
        # Find lines - stright parts of edges out of locks
        line_n = 2
        line_max_mse = 100
        line_mse2 = np.zeros(num_points*2)
        lines = []
        line_cnt2 = []
        i = -1
        for _ in range(line_n, num_points*2 - line_n):
            i += 1

            # Skip locks
            for lock in locks:
                if lock['i0'] <= i+line_n and i-line_n <= lock['i1']:
                    i = lock['i1'] + line_n

            if i >= num_points*2-line_n-1:
                break
            
            line_cum_a = cum_a2[i-line_n : i+line_n+1]
            avg_line = np.average(line_cum_a)
            line_mse = np.mean((line_cum_a - avg_line) ** 2) 
            line_mse2[i] = line_mse
            if line_mse < line_max_mse:
                p1 = np.mean(cnt2[i-line_n : i+1], axis= 0)
                p2 = np.mean(cnt2[i : i+line_n+1], axis= 0)
                lines.append({'i': i, 's': s2[i], 'ca': avg_line, 'p1': p1, 'p2': p2})

                # Line points preview
                for j in range(i-line_n, i+line_n+1):
                    line_cnt2.append(cnt2[j])
        line_cnt2 = np.array(line_cnt2).reshape(-1, 2)

        # Find corners
        corner_min_dca = 60
        corner_max_ds = 100
        corners = []
        corner_cnt2 = []
        for j in range(len(lines)-1):
            # Skip non-corners
            s1_ = lines[j]['s']
            s2_ = lines[j+1]['s']            
            ca1_ = lines[j]['ca']
            ca2_ = lines[j+1]['ca']            
            if abs(ca2_ - ca1_) < corner_min_dca or (s2_ - s1_) > corner_max_ds:
                continue
                
            i1 = lines[j]['i']
            i2 = lines[j+1]['i']

            p1 = lines[j]['p1']
            p2 = lines[j]['p2']
            p3 = lines[j+1]['p1']
            p4 = lines[j+1]['p2']

            v1 = p2-p1
            v2 = p3-p4
            A, B = p1, p4

            matrix = np.array([v1, -v2]).T  # Coefficients for t1, t2
            rhs = B - A  # Right-hand side of the equation
            
            # Solve for t1, t2 (if determinant is nonzero)
            t1, t2 = np.linalg.solve(matrix, rhs)
            corner_cnt = A + t1 * v1  # Compute intersection point
            corner_cnt2.append(corner_cnt)

            # Correct corner in contour cnt2
            distances = np.linalg.norm(cnt2[i1:i2] - corner_cnt, axis=1)
            corner_i = i1 + np.argmin(distances)
            cnt2[corner_i] = corner_cnt
            corner = {'i': corner_i, 's': s2[corner_i], 'ca': (ca1_+ca2_)/2}
            corners.append(corner)

            if len(corners) == 5:
                break
        corner_cnt2 = np.array(corner_cnt2).reshape(-1, 2)

        print(corners)
        print(corner_cnt2)

        # Extract edges
        edges = []
        flat_edges = []
        for edge_id, (c0, c1) in enumerate(zip(corners[:4], corners[1:5])):
            i0, i1 = c0['i'], c1['i']

            e_cnt = cnt2[i0 : i1][0]
            e_s = np.array(s2)[i0 : i1] 
            e_a = np.array(a2)[i0 : i1] 
            e_s -= np.average(e_s)
            e_a -= np.average(e_a)
            e_a_csum = np.cumsum(e_a)
            e_shape = np.column_stack((e_s, e_a_csum))

            # Assign lock to edge
            for lock in locks:
                if i0 <= lock['i'] <= i1:
                    break

            if lock['dir'] == 0:
                flat_edges.append(edge_id)

            edges.append({'e_cnt': e_cnt, 'e_shape': e_shape, 'lock': lock})

        # Rotate edges to have 1st non-flat
        if len(flat_edges):
            e2 = edges + edges
            id = max(flat_edges)+1
            edges = e2[id: id+4]

        piece = {'id': piece_id, 'file': file, 'edges': edges, 'flat_edges': flat_edges}
        pieces.append(piece)

        if plot_piece:
            # Plot piece
            fig, axs = plt.subplots(2, 4, figsize=(10, 5))

            # Plot original points (with duplicates)
            plot_i = -1

            # plot_i += 1
            # axs[0,plot_i].imshow(preview[cnt_y:cnt_y+cnt_h, cnt_x:cnt_x+cnt_w])

            plot_i += 1
            axs[0,plot_i].plot(cnt2[:, 0], cnt2[:, 1], color="blue", marker=".")
            axs[0,plot_i].scatter(lock_cnt2[:, 0],  lock_cnt2[:, 1],  color="red", marker="x")
            axs[0,plot_i].scatter(lock_cnt2_[:, 0], lock_cnt2_[:, 1], color="green", marker="o")
            # axs[0,plot_i].scatter(line_cnt2[:, 0], line_cnt2[:, 1], color="green", marker="x")
            axs[0,plot_i].scatter(corner_cnt2[:, 0], corner_cnt2[:, 1], color="red", marker="x")
            axs[0,plot_i].set_title("contour")

            plot_i += 1
            axs[0,plot_i].plot(s2, cum_a2, color="blue", marker=".") # 
            axs[0,plot_i].scatter(lock_sca2[:, 0],  lock_sca2[:, 1],  color="red", marker="x")
            axs[0,plot_i].scatter(lock_sca2_[:, 0], lock_sca2_[:, 1], color="green", marker="o", s=50)
            axs[0,plot_i].set_title("cum_a2")

            plot_i += 1
            axs[0,plot_i].plot([l['s'] for l in lines],      [l['ca'] for l in lines],   color="blue", marker=".")
            axs[0,plot_i].scatter([c['s'] for c in corners], [c['ca'] for c in corners], color="red",  marker="x")
            axs[0,plot_i].set_title("lines corners")

            # plot_i += 1
            # axs[0,plot_i].plot(s2, line_mse2, color="blue", marker=".")
            # axs[0,plot_i].set_title("line_mse2")

            # plot_i += 1
            # axs[0,plot_i].plot(s2, lock_mse2, color="blue", marker=".")
            # axs[0,plot_i].set_title("lock_mse2")


            for i, edge in enumerate(edges):
                color = 'gray'
                if edge['lock'] == 1:
                    color = 'red'
                elif edge['lock'] == -1:
                    color = 'blue'
                axs[1,i].plot(edge['e_shape'][0], edge['e_shape'][1], color=color)
            
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
            eSA1 = edge['e_SA']

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
                    eSA2 = edge['e_SA']

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
                    e_SA = edge['e_SA']
                    eSA_inv = np.column_stack((e_SA[::-1,0], -e_SA[::-1,1]))
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
 
                        e_SA = edge_adj['e_SA']
                        eSA_inv = np.column_stack((e_SA[::-1,0], -e_SA[::-1,1]))                        
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
                e_cnt = pieces[p_id]['edges'][e_id]['e_cnt']

                for e_id_ in range(4):
                    color='gray'
                    if e_id == e_id_:
                        color = 'green'
                    axs[1,c].plot(e_cnt[:,0], e_cnt[:,1], color=color)
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
