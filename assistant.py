import os
import sys
import math 
import numpy as np
import pandas as pd
import cv2 as cv
import pickle
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from scipy.spatial.distance import euclidean
from scipy.signal import find_peaks
from scipy.signal import medfilt
from scipy.signal import savgol_filter
from shapely.geometry import Point, LineString

local = os.path.dirname(__file__)
os.chdir(local)

def _get_angle(v1, v2):
    # Compute cosine of angle using dot product
    dot_product = np.dot(v1, v2)
    cross_product = np.cross(v1, v2)
    mag_v1 = np.linalg.norm(v1)
    mag_v2 = np.linalg.norm(v2)

    if mag_v1 == 0 or mag_v2 == 0:  # Avoid division by zero
        angle = 0.0
    else:
        cos_angle = dot_product / (mag_v1 * mag_v2)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * np.sign(cross_product)  # Convert to radians
    return angle

def _get_contour(file, num_points, 
                 gauss_blur, eps_ratio, noise_treshold, savgol_kwargs):
    
    img = cv.imread(file)
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
    cnt_rect = cv.boundingRect(cnt.reshape(-1, 1, 2))

    # Compute the signed area using Shoelace formula
    area = 0.5 * np.sum(cnt[:-1, 0] * cnt[1:, 1] - cnt[1:, 0] * cnt[:-1, 1])
    if area > 0:  # True if CW
        cnt = cnt[::-1]  # Reverse order to make CCW
    
    # Prepare preview
    preview = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
    cv.drawContours(preview, [cnt.reshape(-1, 1, 2)], -1, (0, 255, 0), 3)

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
        
        angle = _get_angle(v1, v2)           
        a2[i] = np.degrees(angle)           

    # Cummulated angle
    ca2_raw = np.cumsum(a2, axis = 0)

    # Filter noise
    ca2_filtered = ca2_raw.copy()
    for i in range(1, num_points - 1):  # Ignore first and last point
        dca1 = ca2_filtered[i] - ca2_filtered[i - 1]
        dca2 = ca2_filtered[i+1] - ca2_filtered[i]
        dca_avg = (abs(dca1) + abs(dca2)) / 2
        if np.sign(dca1) * np.sign(dca2) == -1 and dca_avg > noise_treshold:     # Detected outlier
            ca2_filtered[i] = (ca2_filtered[i - 1] + ca2_filtered[i + 1]) / 2  # Replace with mean of neighbors 

    cum_a2 = savgol_filter(ca2_filtered, **savgol_kwargs) 

    return cnt2, s2, cum_a2, preview, cnt_rect, ca2_raw, ca2_filtered

def _get_locks(num_points, cnt2, s2, cum_a2, 
              lock_n_ratio, lock_s_range, lock_cum_a_range, 
              lock_max_mse, lock_edge_max_dca,
              lock_pc_crit, corners = None):

    lock_n = int(num_points/4*lock_n_ratio)

    locks_p0 = []
    locks_pc = []
    locks_p12 = []
    locks_p34 = []

    locks_crit = []
    locks_sca_p0 = []
    locks_sca_p12 = []
    locks_sca_p34 = []

    locks = []
    lock_s_range_ = s2[num_points]/4. * np.array(lock_s_range)
    
    i = lock_n - 1
    for _ in range(lock_n, num_points*2 - lock_n):
        i += 1
        
        # End if out of doubled contour
        if i > num_points*2 - lock_n:
            break

        lock_cum_a = cum_a2[i - lock_n : i + lock_n]
        lock_i_max = np.argmax(lock_cum_a)
        lock_i_min = np.argmin(lock_cum_a)
        i1 = i - lock_n + min(lock_i_min, lock_i_max) 
        i2 = i - lock_n + max(lock_i_min, lock_i_max)  

        if corners:
            skip_i = False
            for corner in corners:
                if i1 <= corner['i'] <= i2:
                    skip_i = True
            if skip_i:
                continue

        d_s = s2[i2] - s2[i1]
        d_cum_a = cum_a2[i2] - cum_a2[i1]
        lock_dir = -np.sign(d_cum_a)

        d_s_ratio =     (abs(d_s) - lock_s_range_[0])/np.diff(lock_s_range_)
        d_cum_a_ratio = (abs(d_cum_a) - lock_cum_a_range[0])/np.diff(lock_cum_a_range)

        # Continue if s or angle range out of range
        if not (0 < d_s_ratio < 1 and 0 < d_cum_a_ratio < 1):
            continue

        # Extend lock range to actual peak            
        for j in range(i1-1, i1-lock_n, -1):
            if j < 0: break 
            if cum_a2[j]*lock_dir > cum_a2[j+1]*lock_dir:
                i1 = j
            else: break
        
        for j in range(i2+1, i2+lock_n, 1):  
            if j > num_points*2: break  
            if cum_a2[j-1]*lock_dir > cum_a2[j]*lock_dir:
                i2 = j
            else: break

        # Check circularity of lock by mse
        x = s2[i1 : i2]
        y = cum_a2[i1 : i2]
        a, b = np.polyfit(x, y, 1)  # 1 means linear fit
        line = a*x + b
        mse = np.mean((y - line) ** 2) 
        mse_ratio =  mse/lock_max_mse

        # Reject potential locks not matching to peak-to-peak line
        if not (mse_ratio < 1):
            continue

        # Move index i to middle point of lock and calc center of lock - pc
        i =   int((i1 + i2) / 2)
            
        # Extend lock range to point on edge 
        i3, i4 = i1, i2
        for j in range(i1-1, i1-lock_n, -1):
            if j < 0: break 
            if abs(cum_a2[j] - cum_a2[i]) < lock_edge_max_dca:
                i3 = j
                break
        
        for j in range(i2+1, i2+lock_n, 1):  
            if j > num_points*2: break  
            if abs(cum_a2[j] - cum_a2[i]) < lock_edge_max_dca:
                i4 = j
                break

        p0 = cnt2[i]
        p1, p2, p3, p4 = cnt2[i1], cnt2[i2], cnt2[i3], cnt2[i4]
        pc = np.mean([p0, p1, p2], axis=0)            

        # Reject potential lock if overlaps previous locks
        overlap = False
        for lock_prev in locks:
            i3_prev = lock_prev['i'][0] 
            i4_prev = lock_prev['i'][4]
            pc_prev = lock_prev['pc']

            # Check overlaping locks (not repeated locks)
            d_pc_ratio = np.linalg.norm(pc-pc_prev)/lock_pc_crit
            if  d_pc_ratio > 1:
                if (i3_prev <= i3 - num_points <= i4_prev or 
                    i3_prev <= i4 - num_points <= i4_prev or 
                    i3_prev <= i3 <= i4_prev or 
                    i3_prev <= i4 <= i4_prev):
                    overlap = True
                    i = i4
        if overlap: continue

        locks_pc.append( pc)
        locks_p0.append( p0)
        locks_p12.append(p1)
        locks_p12.append(p2)
        locks_p34.append(p3)
        locks_p34.append(p4)

        locks_sca_p0.append( [s2[i],  cum_a2[i]])
        locks_sca_p12.append([s2[i1], cum_a2[i1]])
        locks_sca_p12.append([s2[i2], cum_a2[i2]])
        locks_sca_p34.append([s2[i3], cum_a2[i3]])
        locks_sca_p34.append([s2[i4], cum_a2[i4]])

        locks_crit.append([s2[i], d_s_ratio, d_cum_a_ratio, mse_ratio])         
        lock = {'i': [i3, i1, i, i2, i4], 'dir': lock_dir, 'pc': pc}
        locks.append(lock)                   

        # Jump to next lock
        i += lock_n

    locks_pc =  np.array(locks_pc).reshape(-1, 2)
    locks_p0 =  np.array(locks_p0).reshape(-1, 2)
    locks_p12 = np.array(locks_p12).reshape(-1, 2)
    locks_p34 = np.array(locks_p34).reshape(-1, 2)
    locks_p = (locks_pc, locks_p0, locks_p12, locks_p34)

    locks_sca_p0 =  np.array(locks_sca_p0).reshape(-1, 2)
    locks_sca_p12 = np.array(locks_sca_p12).reshape(-1, 2)      
    locks_sca_p34 = np.array(locks_sca_p34).reshape(-1, 2)      
    locks_sca = (locks_sca_p0, locks_sca_p12, locks_sca_p34) 

    locks_crit =    np.array(locks_crit).reshape(-1, 4)

    return locks, locks_p, locks_sca, locks_crit

def _get_lines(num_points, cnt2, s2, cum_a2, locks, line_n_ratio, line_max_mse):
    # Find lines - stright parts of edges out of locks
    line_n = int(num_points/4*line_n_ratio)
    lines = []
    lines_cnt2 = []
    lines_crit = []

    # Start after first lock
    i = locks[0]['i'][-1]
    for _ in range(line_n, num_points*2 - line_n):
        i += 1

        # Skip locks
        for lock in locks:
            if lock['i'][0] <= i <= lock['i'][-1]:
                i = lock['i'][-1]
                break

            if lock['i'][0] <= i - num_points <= lock['i'][-1]:
                i = lock['i'][-1] + num_points
                break

        if i >= num_points*2-line_n-1: break
        
        line_cum_a = cum_a2[i-line_n : i+line_n+1]
        avg_line = np.average(line_cum_a)
        line_mse_ratio = np.mean((line_cum_a - avg_line) ** 2) / line_max_mse

        if line_mse_ratio < 1:
            p1 = np.mean(cnt2[i-line_n : i+1], axis= 0)
            p2 = np.mean(cnt2[i : i+line_n+1], axis= 0)
            lines.append({'i': i, 's': s2[i], 'ca': avg_line, 'p1': p1, 'p2': p2, 'mser': line_mse_ratio})
            lines_cnt2.append(cnt2[i])
            lines_crit.append([s2[i], line_mse_ratio, cum_a2[i]])

    lines_cnt2 = np.array(lines_cnt2).reshape(-1, 2)
    lines_crit = np.array(lines_crit).reshape(-1, 3)

    return lines, lines_cnt2, lines_crit

def _select_corners(num_points, cnt2, s2, cum_a2):

    fig, ax = plt.subplots(figsize=(6, 6))

    # Set plot limits and aspect ratio
    ax.plot(cnt2[:, 0], cnt2[:, 1], color="blue", marker=".")
    ax.set_aspect("equal")
    ax.set_title("Click on the 4 real corners of the piece")
    plt.draw()

    corners_cnt2 = plt.ginput(4, timeout=0)  
    corners_cnt2 = np.array(corners_cnt2).reshape(-1, 2)
    plt.scatter(*zip(*corners_cnt2), color='red', marker='x', s=100, label="Selected Corners")
    
    plt.close()

    corners_i = []
    for corner_cnt in corners_cnt2:
        # Correct corner in contour cnt2
        distances = np.linalg.norm(cnt2[:num_points] - corner_cnt, axis=1)
        corner_i = np.argmin(distances)        
        corners_i.append(corner_i)
        cnt2[corner_i] = corner_cnt 
    corners_i.sort()
    corners_i.append(corners_i[-1] + num_points)

    corners = []
    for corner_i in corners_i:       
        corner = {'i': corner_i, 's': s2[corner_i], 'ca': cum_a2[corner_i]}
        corners.append(corner)

    return cnt2, corners, corners_cnt2, []

def _get_corners(num_points, cnt2, s2, cum_a2, lines, corner_max_ds, corner_cum_a_range):
    # Find corners
    corners = []
    corners_cnt2 = []
    corners_crit = []
    for j in range(len(lines)-1):

        # Check min distance between lines
        p1 = lines[j+1]['p1']
        p2 = lines[j]['p2']
        d_s = np.linalg.norm(p2-p1)

        d_s_ratio = d_s/corner_max_ds
        d_cum_a = lines[j+1]['ca'] - lines[j]['ca']
        d_cum_a_ratio = (d_cum_a - corner_cum_a_range[0])/ np.diff(corner_cum_a_range)
        
        if not (d_s_ratio < 1 and 0 <= d_cum_a_ratio <= 1):
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
        corners_cnt2.append(corner_cnt)

        # Correct corner in contour cnt2
        distances = np.linalg.norm(cnt2[i1:i2] - corner_cnt, axis=1)
        corner_i = i1 + np.argmin(distances)
        cnt2[corner_i] = corner_cnt

        s = (lines[j+1]['s'] + lines[j]['s'])/2
        ca_avg = (lines[j+1]['ca'] + lines[j]['ca'])/2
        line_mse_ratio = lines[j]['mser']
        corners_crit.append([s, d_s_ratio, d_cum_a_ratio, line_mse_ratio, ca_avg])

        corner = {'i': corner_i, 's': s2[corner_i], 'ca': ca_avg}
        corners.append(corner)

        if len(corners) == 5:
            break

    corners_cnt2 = np.array(corners_cnt2).reshape(-1, 2)
    corners_crit = np.array(corners_crit).reshape(-1, 5)

    return cnt2, corners, corners_cnt2, None

def _get_edges(num_points, cnt2, s2, cum_a2, locks, corners):
    # Extract edges
    edges = []
    flat_edges = []
    for edge_i, (c1, c2) in enumerate(zip(corners[:4], corners[1:5])):
        i1, i2 = c1['i'], c2['i']


        e_s = s2[i1 : i2+1] - np.mean(s2[i1 : i2+1], axis=0)
        e_ca = cum_a2[i1 : i2+1] - np.mean(cum_a2[i1 : i2+1], axis=0)
        e_sca = np.column_stack([e_s, e_ca])

        p1, p2 = cnt2[i1], cnt2[i2]
        e_len = np.linalg.norm(p2-p1)

        # Compute rotation matrix
        v1 = p2 - p1            
        theta = _get_angle(v1, np.array([1,0])) 
        R = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta),  np.cos(theta)]])
        e_cnt1 = cnt2[i1]
        e_cnt = cnt2[i1 : i2+1] - e_cnt1
        e_cnt = np.dot(e_cnt, R.T)

        # Assign lock to edge
        lock = None
        for lock_ in locks:
            if (i1 <= lock_['i'][2] <= i2) or (i1 <= lock_['i'][2] + num_points <= i2):
                lock = lock_
                break

        l_dir, l_pc = 0, None
        if lock:
            # Get lock dir and rotated pc
            l_dir = lock['dir']
            l_pc = lock['pc']- e_cnt1
            l_pc = np.dot(l_pc, R.T)
        else:
            flat_edges.append(edge_i)

        edges.append({'p1': p1, 'p2': p2, 'e_cnt': e_cnt, 'e_len': e_len, 'l_dir': l_dir, 'l_pc': l_pc}) #, 'e_sca': e_sca

    # Rotate edges to have 1st non-flat
    if len(flat_edges):
        e2 = edges + edges
        edge_i = max(flat_edges)
        edges = e2[edge_i+1 : edge_i + 5]

    return edges, flat_edges

def _get_p_e(puzzle, r, c):
    M, N = puzzle.shape

    if not (0 <= r < M and 0 <= c < N):
        return None, None
    
    piece_id = puzzle[r, c]
    
    if piece_id >= 0:
        p_id = math.floor(piece_id)
        diff = round((piece_id-p_id)*10,0)
        e_id = int(diff)-1
        return p_id, e_id
    else:
        return None, None

def _offset_edge(e_id, e_off):
    e_id += e_off
    if e_id > 3:
        e_id -= 4
    if e_id < 0:
        e_id += 4
    return e_id

def get_pieces(  pieces_dir = None, files_range = None,
                pieces = {}, pieces_file = None, p_set = 0, select_corners = [], locks_range = [2,4],
                plot_piece = False, start = 1, 
                gauss_blur = (5,5), 
                eps_ratio = .0004,
                num_points = 200,

                noise_treshold = 60,
                savgol_kwargs = {'window_length': 5, 'polyorder': 2},

                lock_n_ratio = .5,
                lock_s_range = [.1, .5], # % part of avg edge length
                lock_cum_a_range = [180, 300], # lock angular range [deg]
                lock_max_mse = 350, # circularity crit.
                lock_edge_max_dca = 45, #[deg]
                lock_pc_crit = 100,

                line_n_ratio = .05,
                line_max_mse = 100,

                corner_max_ds = 300,               
                corner_cum_a_range = [-135, -45],
               ):

    local = os.path.dirname(__file__)
    os.chdir(local)

    if not pieces_dir:
        pieces_dir = f'pieces_{p_set}'

    path = os.path.join(local, pieces_dir)
    filenames = os.listdir(path)
    filenames.sort()   
    if files_range:
        filenames = filenames[files_range[0]-1 : files_range[1]]

    print(f"""  
                Parameters:
                ----------------------------------
                gauss_blur = {gauss_blur}
                eps_ratio = {eps_ratio}
                num_points = {num_points}

                noise_treshold = {noise_treshold}
                savgol_kwargs = {savgol_kwargs}

                lock_n_ratio = {lock_n_ratio}
                lock_s_range = {lock_s_range}
                lock_cum_a_range = {lock_cum_a_range}
                lock_max_mse = {lock_max_mse}
                lock_edge_max_dca = {lock_edge_max_dca}
                lock_pc_crit = {lock_pc_crit}

                line_n_ratio = {line_n_ratio}
                line_max_mse = {line_max_mse}

                corner_max_ds = {corner_max_ds}       
                corner_cum_a_range = {corner_cum_a_range}
          """)
    pieces_len = len(filenames)

    for piece_i, file in enumerate(filenames, start = start):
        plot_piece_i = plot_piece
        piece_id = p_set*1000 + piece_i
        file = os.path.join(pieces_dir,file)
                
        cnt2, s2, cum_a2, preview, cnt_rect, cum_a2_raw, ca2_filtered = _get_contour(file, num_points, gauss_blur, eps_ratio, noise_treshold, savgol_kwargs)
        # cnt_x, cnt_y, cnt_w, cnt_h = cnt_rect

        get_corners = False
        if piece_i in select_corners:
            get_corners = True

        for try_i in range(2):
            corners = None
            if try_i == 1 or get_corners:
                cnt2, corners, corners_cnt2, corners_crit = _select_corners(num_points, cnt2, s2, cum_a2)
                plot_piece_i = True  

            locks, locks_p, locks_sca, locks_crit = _get_locks(num_points, cnt2, s2, cum_a2, 
                                                            lock_n_ratio, lock_s_range, lock_cum_a_range, 
                                                            lock_max_mse, lock_edge_max_dca, lock_pc_crit, corners)
        
            locks_pc, locks_p0, locks_p12, locks_p34 = locks_p
            locks_sca_p0, locks_sca_p12, locks_sca_p34 = locks_sca    
            lines, lines_cnt2, lines_crit  = _get_lines(num_points, cnt2, s2, cum_a2, locks, line_n_ratio, line_max_mse)

            if corners is None:
                cnt2, corners, corners_cnt2, corners_crit = _get_corners(num_points, cnt2, s2, cum_a2, lines, corner_max_ds, corner_cum_a_range)
            
            edges, flat_edges = _get_edges(num_points, cnt2, s2, cum_a2, locks, corners)

            edge_locks = 0
            for edge in edges:
                edge_locks += abs(edge['l_dir'])
            print(f'{piece_i}/{pieces_len} - {file}, flat_edges: {len(flat_edges)}, edge_locks {edge_locks}')

            # If enough locks, skip manual corners selection
            if locks_range[0] <= edge_locks <= locks_range[1]:
                break


        piece = {'file': file, 'cnt': cnt2[:num_points], 'edges': edges, 'flat_edges': flat_edges, 'locks_pc': locks_pc[:4-len(flat_edges)], 'corners_cnt': corners_cnt2[:4]}
        pieces.update({piece_id: piece})

        if plot_piece_i:
            # Plot piece
            fig, axs = plt.subplots(2, 4, figsize=(12, 6))
            fig.suptitle(f'Piece {piece_id}, file {file}')

            # Plot original points (with duplicates)
            plot_i = -1

            # plot_i += 1
            # axs[0,plot_i].imshow(preview[cnt_y:cnt_y+cnt_h, cnt_x:cnt_x+cnt_w])

            plot_i += 1
            axs[0,plot_i].plot(cnt2[:, 0], cnt2[:, 1], color="blue", marker=".")
            axs[0,plot_i].scatter(locks_pc[:, 0],  locks_pc[:, 1],  color="red", marker="+")
            axs[0,plot_i].scatter(locks_p0[:, 0],  locks_p0[:, 1],  color="red", marker="o")
            axs[0,plot_i].scatter(locks_p12[:, 0], locks_p12[:, 1], color="pink", marker="o")
            axs[0,plot_i].scatter(locks_p34[:, 0], locks_p34[:, 1], color="green", marker="o")
            axs[0,plot_i].scatter(corners_cnt2[:, 0], corners_cnt2[:, 1], color="red", marker="x")
            axs[0,plot_i].scatter(lines_cnt2[:, 0], lines_cnt2[:, 1], color="orange", marker="+", s=100)
            axs[0,plot_i].set_title("contour")

            plot_i += 1
            axs[0,plot_i].plot(s2,cum_a2_raw, color="lightgray", marker=".") 
            axs[0,plot_i].plot(s2,ca2_filtered, color="lightblue", marker=".") 
            axs[0,plot_i].plot(s2,cum_a2, color="blue") 
            axs[0,plot_i].scatter(locks_sca_p0[:, 0], locks_sca_p0[:, 1],  color="red", marker="o", s=50)   #  
            axs[0,plot_i].scatter(locks_sca_p12[:, 0],locks_sca_p12[:, 1], color="pink", marker="o", s=50) # 
            axs[0,plot_i].scatter(locks_sca_p34[:, 0],locks_sca_p34[:, 1], color="green", marker="o", s=50) # 
            axs[0,plot_i].scatter(lines_crit[:, 0],   lines_crit[:, -1],   color="orange", marker="+", s=100)
            if corners_crit:
                axs[0,plot_i].scatter(corners_crit[:, 0], corners_crit[:, -1], color="red", marker="x", s=100)
            axs[0,plot_i].set_title("locks")

            plot_i += 1
            axs[0,plot_i].scatter(locks_crit[:, 0],  locks_crit[:, 1],  color="blue",  label='ds')   #  
            axs[0,plot_i].scatter(locks_crit[:, 0],  locks_crit[:, 2],  color="red",   label='dcuma' )   #  
            axs[0,plot_i].scatter(locks_crit[:, 0],  locks_crit[:, 3],  color="green", label='mse' )   #  
            axs[0,plot_i].set_title("locks crit (ds, dca, mse)")
            axs[0,plot_i].set_ylim(0,1)
            axs[0,plot_i].legend()

            plot_i += 1 
            axs[0,plot_i].scatter(lines_crit[:, 0],   lines_crit[:, 1],   color="lightblue", label='l_mser')
            if corners_crit:
                axs[0,plot_i].scatter(corners_crit[:, 0], corners_crit[:, 1], color="red",   label='c_ds')
                axs[0,plot_i].scatter(corners_crit[:, 0], corners_crit[:, 2], color="green", label='c_dcuma')
                axs[0,plot_i].scatter(corners_crit[:, 0], corners_crit[:, 3], color="blue", label='c_mser')
            axs[0,plot_i].set_title("line crit (l_mse, c_ds, c_dcuma, c_mser)")
            axs[0,plot_i].legend()

            for plot_i, edge in enumerate(edges):
                lock_dir = edge['l_dir']
                color = 'gray'
                if lock_dir == 1:
                    color = 'red'
                elif lock_dir == -1:
                    color = 'blue'
                axs[1,plot_i].plot(edge['e_cnt'][:, 0], edge['e_cnt'][:, 1], color=color)
                axs[1,plot_i].set_title(f"edge {plot_i} {lock_dir}")
            
            # Show the plots
            plt.show()
    
    # Write pieces to file
    if not pieces_file:
        pieces_file = f'pieces_{p_set}.pkl'
    
    with open(pieces_file, "wb") as f:
        pickle.dump(pieces, f)

    return pieces

def load_pieces( pieces = {}, pieces_file = None, p_set = 0, plot_pieces = True, x_sign = 1, y_sign = 1):

    local = os.path.dirname(__file__)
    os.chdir(local)

    if not pieces_file:
        pieces_file = f'pieces_{p_set}.pkl'

    # Read pieces from file
    if not os.path.isfile(pieces_file):
        raise ValueError(f'Pieces file {pieces_file} not found')
    
    with open(pieces_file, "rb") as f:
        new_pieces = pickle.load(f)   

    print(f'Loaded {len(new_pieces)} pieces from {pieces_file}')
    pieces.update(new_pieces)

    if plot_pieces:
        n = 10
        m = math.ceil(len(pieces) / n)
        m = max(2,m)
        fig, axs = plt.subplots(m, n) # figsize = plots_shape 
        fig.suptitle(f'File {pieces_file}')
        r, c = 0, -1

        for p_id, piece in pieces.items():
            
            c += 1
            if c == n:
                r += 1
                c = 0
            if r == m:
                break

            cnt = piece['cnt']                        
            locks_pc = piece['locks_pc']                        
            corners_cnt = piece['corners_cnt']   
            file = piece['file']                     
            axs[r,c].plot(   x_sign*cnt[:,0],         y_sign*cnt[:,1],         color='blue')
            axs[r,c].scatter(x_sign*locks_pc[:,0],    y_sign*locks_pc[:,1],    color='red', marker='+')
            axs[r,c].scatter(x_sign*corners_cnt[:,0], y_sign*corners_cnt[:,1], color='red', marker='x')
            axs[r,c].set_title(f'{p_id}, {file[-10:-4]}', pad=0)

            axs[r,c].set_xticks([])  # Remove x-axis ticks
            axs[r,c].set_yticks([])  # Remove y-axis ticks
            axs[r,c].set_xticklabels([])  # Remove x-axis labels
            axs[r,c].set_yticklabels([])  # Remove y-axis labels            
        plt.show()       

    return pieces

def init_puzzle(puzzle = (2, 2), pieces = [], p_file = None, p_id = None, p_pos = (1, 1), p_edge = 1):
    """init puzzle

    Args:
        puzzle (tuple, optional): size tuple or puzzle array. Defaults to (M, N).
        pieces (list, optional): list of pieces. Defaults to [].
        p_file (_type_, optional): initial piece file. Defaults to None.
        p_id (_type_, optional): initial piece id. Defaults to None.
        p_pos (tuple, optional): initial piece position (starting from (1,1)). Defaults to (1, 1).
        p_edge (int, optional): initial piece edge id (starting from 1). Defaults to 1.

    Returns:
        _type_: _description_
    """
    if type(puzzle) == tuple:
        puzzle = -np.ones((puzzle[0], puzzle[1]), dtype=float)
    
    piece_found = False
    if not p_id and p_file:
        for p_id, piece in pieces.items():
            if p_file in piece['file']:
                piece_found = True
                break
    
    if not piece_found:
        raise ValueError(f'Piece {p_file} not found')
    
    # Set initial piece
    piece_id = p_id + p_edge/10
    puzzle[p_pos[0]-1, p_pos[1]-1] = piece_id
    print(f'Initial piece {piece_id}')

    with open("puzzle.pkl", "wb") as f:
        pickle.dump(puzzle, f)

    return puzzle

def match_frame(puzzle, pieces):

    puzzle = match_puzzle(puzzle, pieces, from_pos = (1,1), to_pos = (M,1))
    puzzle = match_puzzle(puzzle, pieces, from_pos = (M,1), to_pos = (M,N))
    puzzle = match_puzzle(puzzle, pieces, from_pos = (M,N), to_pos = (1,N))
    puzzle = match_puzzle(puzzle, pieces, from_pos = (1,N), to_pos = (1,1))

    return puzzle

def match_puzzle(puzzle, pieces, from_pos = (2,1), to_pos = (3,1), 
                 lock_pc_crit = .2,
                 edge_len_crit = .2,
                 edge_angle_crit = None, # test in progress
                 edge_dist_crit = 500.,

                 candidates_diff_crit = 10,
                 plot_candidates = True, confirm = True, verbose = True
                 ):
    """
        Default orientation of corner piece
            4
            -----
            |     |
        1   >     | 3
            |     |
            --^--
            2

        Default orientation of border piece
            4
            -----
            |     |
        1  >     < 3
            |     |
            --^--
            2      

        Piece _p is checked if matches on e1 to p1, then on e2 to p2 (not if in 1st column)

        p   |  p   |  p1 
            |      |
                    --e1--
            |      |  
        p   |  p2 e2  _p
            |      |

        Reqired flat edges

        2 1 1 .. 1 1 2
        1 0 0 .. 0 0 1
        ..
        1 0 0 .. 0 0 1
        2 1 1 .. 1 1 2

    """
    M, N = puzzle.shape
    puzzle_corners = np.zeros((M+1, N+1, 2), dtype=float)

    # Use rows and cols starting from 0
    row0, col0 = from_pos
    row1, col1 = to_pos
    row0, col0, row1, col1 = row0-1, col0-1, row1-1, col1-1

    d_r = +1 if row1 >= row0 else -1
    d_c = +1 if col1 >= col0 else -1

    # Set offset of adjacent horizontal edge
    e1_off = 2 if d_r == 1 else 0
    e2_off = -d_c

    _e1_off = 2 if d_r == -1 else 0  
    _e2_off = d_c

    # Set rows and cols
    rows = range(row0, row1 + d_r, d_r)
    p1_off = (-d_r, 0)

    cols = range(col0, col1 + d_c, d_c)
    p2_off = (0,-d_c)

    # required number of flat edges
    req_piece_type = np.zeros((M, N), dtype=int)
    req_piece_type[ 0, :] = 1
    req_piece_type[-1, :] = 1
    req_piece_type[ :, 0] = 1
    req_piece_type[ :,-1] = 1
    req_piece_type[ 0, 0] = 2
    req_piece_type[ 0,-1] = 2
    req_piece_type[-1,-1] = 2
    req_piece_type[-1, 0] = 2    

    print(f'Start from piece {from_pos} to {to_pos}')

    for col in cols:
        for row in rows:
            
            # Skip first or matched piece
            piece_id = puzzle[row, col] 
            if piece_id >= 0:
                if verbose:
                    print(f'Skip piece {piece_id} already matched')
                continue            

            # Get verticaly reference piece - p1
            r1, c1 = row + p1_off[0], col + p1_off[1]
            p1_id, e1_id = _get_p_e(puzzle, r1, c1)

            # Get horizontaly reference piece - p2
            r2, c2 = row + p2_off[0], col + p2_off[1]
            p2_id, e2_id = _get_p_e(puzzle, r2, c2)
            
            print(f'\nProcessing piece ({row+1},{col+1})')
            
            e1_len, e1_cnt_inv, e1_linestr, l1_dir, l1_pc_inv = None, None, None, None, None
            if p1_id is not None:

                # Get opposite edge to matched previously in top-bottom or bottom-top match
                e1_id = _offset_edge(e1_id, e1_off)

                print(f'Vertically adjacent piece ({r1+1}, {c1+1}) {p1_id}.{e1_id+1}')
                piece1 = pieces[p1_id]
                edge1 = piece1['edges'][e1_id]
                l1_dir = edge1.get('l_dir')
                l1_pc = edge1.get('l_pc')
                e1_cnt = edge1['e_cnt']
                e1_len = edge1['e_len']

                e1_cnt_inv = np.column_stack((e1_len - e1_cnt[:,0], -e1_cnt[:,1]))[::-1]
                e1_linestr = LineString(e1_cnt_inv)

                if l1_pc is not None:
                    l1_pc_inv = [e1_len - l1_pc[0], -l1_pc[1]]                           
                
            e2_len, e2_cnt_inv, e2_linestr, l2_dir, l2_pc_inv = None, None, None, None, None
            if p2_id is not None:

                # Get opposite edge to matched previously in horizontal match
                e2_id = _offset_edge(e2_id, e2_off)

                print(f'Horizontally adjacent piece ({r2+1}, {c2+1}) {p2_id}.{e2_id+1}')

                piece2 = pieces[p2_id]
                edge2 = piece2['edges'][e2_id]
                e2_cnt = edge2['e_cnt']
                e2_len = edge2['e_len']
                l2_dir = edge2['l_dir']
                l2_pc = edge2['l_pc']

                e2_cnt_inv = np.column_stack((e2_len - e2_cnt[:,0], -e2_cnt[:,1]))[::-1]
                e2_linestr = LineString(e2_cnt_inv)

                if l2_pc is not None:
                    l2_pc_inv = [e2_cnt[-1,0] - l2_pc[0], -l2_pc[1]]

            # Check all available pieces
            candidates = []
            for _p_id, _piece in pieces.items():
                
                file = _piece['file']                

                # Skip pieces already used
                if np.isin(puzzle, [_p_id+.1, _p_id+.2, _p_id+.3, _p_id+.4]).any():
                    if verbose==2:
                        print(f'Skip piece {_p_id} - already used')
                    continue
                
                # Try all edges of _piece - _e1_id is a TOP edge of _piece
                for _e_id in range(4):     

                    piece_id = _p_id + (_e_id+1)/10  

                    # Get 1st edge to match verticaly
                    _e1_id = _offset_edge(_e_id, _e1_off)                    
                    _edge1 = _piece['edges'][_e1_id]
                    _e1_cnt = _edge1['e_cnt']
                    _l1_dir = _edge1['l_dir']
                    _l1_pc = _edge1['l_pc']

                    # Get 2nd edge to match horizontaly
                    _e2_id = _offset_edge(_e_id, _e2_off)                                                   
                    _edge2 =  _piece['edges'][_e2_id]
                    _e2_cnt = _edge2['e_cnt']    
                    _l2_dir = _edge2['l_dir']     
                    _l2_pc =  _edge2['l_pc']   

                    _e1_len = np.linalg.norm(_e1_cnt[-1]-_e1_cnt[0])
                    _e2_len = np.linalg.norm(_e2_cnt[-1]-_e2_cnt[0])

                    # Skip piece with wrong number of flat edges
                    piece_type = len(_piece['flat_edges'])
                    if piece_type != req_piece_type[row, col]:
                        if verbose:
                            print(f'Skip piece {piece_id} with wrong number of flat edges ({piece_type} != {req_piece_type[row, col]})')
                        break

                    # Skip not flat horiz. external edge
                    if (col == 0 or col == N-1):
                        
                        if col == 0:
                            _e3_id = _offset_edge(_e_id, 1)
                        if col == N-1:
                            _e3_id = _offset_edge(_e_id, -1)

                        _edge3 = _piece['edges'][_e3_id]
                        _l3_dir = _edge3['l_dir']  

                        if _l3_dir != 0:
                            if verbose:
                                print(f'Skip edge {piece_id} with not flat external edge {_e3_id+1}')
                            continue                    
                    
                    # Skip not flat vert. external edge
                    if (row == 0 or row == M-1):
                        
                        if row == 0:
                            _e3_id = _offset_edge(_e_id, 0)
                        if row == M-1:
                            _e3_id = _offset_edge(_e_id, 2)

                        _edge3 = _piece['edges'][_e3_id]
                        _l3_dir = _edge3['l_dir']  

                        if _l3_dir != 0:
                            if verbose:
                                print(f'Skip edge {piece_id} with not flat external edge {_e3_id+1}')
                            continue


                    # Skip not-matching lock directions on edges
                    if _l1_dir and l1_dir and _l1_dir != -l1_dir:
                        if verbose:
                            print(f'Skip edge {piece_id} - not-matching vertically')
                        continue

                    if _l2_dir and l2_dir and _l2_dir != -l2_dir:
                        if verbose:
                            print(f'Skip edge {piece_id} - not-matching horizontally')
                        continue


                    # Skip edges with not matching length
                    if type(edge_len_crit) is float:
                        e1_len_dist = 0
                        if e1_len and _e1_len:
                            e1_len_dist = np.abs(_e1_len - e1_len)/(e1_len*edge_len_crit)

                        e2_len_dist = 0
                        if e2_len and _e2_len:
                            e2_len_dist = np.abs(_e2_len - e2_len)/(e2_len*edge_len_crit)

                        if e1_len_dist > 1 or e2_len_dist > 1:
                            if verbose:
                                print(f'Skip edge {piece_id} with not matching edge length', round(e1_len_dist,2), round(e2_len_dist,2))
                            continue

                    # Skip pieces with not matching lock center
                    l1_pc_dist, l2_pc_dist = 0, 0
                    if type(lock_pc_crit) is float:
                        l1_pc_dist = 0
                        if _l1_pc is not None and l1_pc_inv is not None and e1_len:
                            l1_pc_dist = np.linalg.norm(_l1_pc - l1_pc_inv)/(e1_len*lock_pc_crit) 
                            
                        l2_pc_dist = 0
                        if _l2_pc is not None and l2_pc_inv is not None and e2_len:
                            l2_pc_dist = np.linalg.norm(_l2_pc-l2_pc_inv)/(e2_len*lock_pc_crit)

                        if l1_pc_dist > 1 or l2_pc_dist > 1:
                            if verbose:
                                print(f'Skip edge {piece_id} with not matching lock center', round(l1_pc_dist,2), round(l2_pc_dist,2))
                            continue

                    # Skip pieces with not matching edges angle
                    if type(edge_angle_crit) is float and row != row0 and col != col0:

                        p0 = puzzle_corners[row, col]
                        p1 = puzzle_corners[row, col+d_c]
                        p2 = puzzle_corners[row+d_r, col]

                        v1 = p1 - p0
                        v2 = p2 - p0          
                        e12_angle = abs(_get_angle(v1, v2))

                        _e1_p1 = _edge1['p1']
                        _e1_p2 = _edge1['p2']
                        _e2_p1 = _edge2['p1']
                        _e2_p2 = _edge2['p2']

                        _v1 = _e1_p2 - _e1_p1  
                        _v2 = _e2_p2 - _e2_p1          
                        _e12_angle = _get_angle(_v1, _v2)

                        if e12_angle > 0 and np.abs(_e12_angle - e12_angle)/(e12_angle*edge_angle_crit) > 1:
                            if verbose:
                                print(f'Skip piece {piece_id} with not matching edge angle', round(_e12_angle,2), round(e12_angle,2))
                            continue
                     
                    # Measure how firmly pieces match to each other
                    e1_dist, e2_dist = 0, 0
                    if type(edge_dist_crit) == float:
                        
                        if e1_linestr:
                            _e1_dist = [Point(p).distance(e1_linestr) for p in _e1_cnt]                    
                            e1_dist = np.sum(_e1_dist) / edge_dist_crit
                    
                        if e2_linestr:
                            _e2_dist = [Point(p).distance(e2_linestr) for p in _e2_cnt]                    
                            e2_dist = np.sum(_e2_dist) / edge_dist_crit
                        
                        if e1_dist > 1 or e2_dist > 1:
                            if verbose:
                                print(f'Skip edge {piece_id} above edge_dist_crit', e1_dist, e2_dist)
                            continue

                    # Append candidate                    
                    total_dist = (e1_dist**2 + e2_dist**2)**.5
                    candidate = {'p_id': _p_id, 'e_id': _e_id, 'p_file': _piece['file'], 
                                'dist': total_dist, 'dist_crit': np.array([l1_pc_dist, e1_dist, l2_pc_dist, e2_dist]).round(2),
                                }
                    candidates.append(candidate)
                                        
            if len(candidates) == 0:
                print(f'No candidates found for ({row+1},{col+1})')  
                c_id_str = input(f'Enter -1 to skip or -2 to exit', ).strip()

                if c_id_str == '-1':
                    break

                if c_id_str == '-2':
                    # terminate and save puzzle
                    write_puzzle(puzzle)
                    exit()
            
            # Sort candidates            
            candidates = sorted(candidates, key=lambda c: c["dist"])            
            max_cand = min(4, len(candidates))

            # List candidates
            print(f'For loc ({row+1},{col+1}) (p1 {p1_id}.{e1_id}) found {len(candidates)} candidates:')
            
            recommended = ''
            if len(candidates) > 1:
                c0_dist = candidates[0]['dist']
                c1_dist = candidates[1]['dist']

                if c0_dist > 0 and (c1_dist - c0_dist)/c0_dist < candidates_diff_crit:
                    recommended = ' - RECOMMENDED'
            
            for i, candidate in enumerate(candidates[:max_cand]):
                p_id = candidate['p_id']
                e_id = candidate['e_id']      
                file = candidate['p_file']
                dist_crit = candidate['dist_crit']

                # if 2 first pieces have sufficiently big difference in dist_crit, 1st is recommended
                print(f'({i}) - {p_id}.{e_id+1}, file={file}, dist_crit={dist_crit}{recommended}')
                recommended = ''

            # Plot candidates
            if plot_candidates:
                fig, axs = plt.subplots(2, max(max_cand,2), figsize=(10, 5))                
                for c, candidate in enumerate(candidates[:max_cand]):

                    _p_id = candidate['p_id']
                    _e_id = candidate['e_id']

                    # Get candidte edge to match verticaly
                    _e1_id = _offset_edge(_e_id, _e1_off)
                    _edge1 = pieces[_p_id]['edges'][_e1_id]
                    _e1_cnt = _edge1['e_cnt']
                    _l1_dir = _edge1['l_dir']
                    _l1_pc =  _edge1['l_pc']

                    # Get candidte edge to match horizontaly
                    _e2_id = _offset_edge(_e_id, _e2_off)
                    _edge2 = pieces[_p_id]['edges'][_e2_id]
                    _e2_cnt = _edge2['e_cnt']
                    _l2_dir = _edge2['l_dir']
                    _l2_pc =  _edge2['l_pc']                            

                    if e1_cnt_inv is not None:
                        axs[0,c].plot(e1_cnt_inv[:,0], e1_cnt_inv[:,1], color='blue', marker='.')
                    if l1_pc_inv is not None:
                        axs[0,c].scatter(l1_pc_inv[0], l1_pc_inv[1], color='blue', marker='+')
                    if isinstance(_e1_cnt, np.ndarray) and _e1_cnt.ndim == 2 :
                        axs[0,c].plot(_e1_cnt[:,0], _e1_cnt[:,1], color='red', marker='.')
                    if _l1_pc is not None:
                        axs[0,c].scatter(_l1_pc[0], _l1_pc[1], color='red', marker='+')
                        axs[0,c].set_title(f'{_p_id}:{_e_id+1}')
                        axs[0,c].axis('equal')

                    if e2_cnt_inv is not None:
                        axs[1,c].plot(e2_cnt_inv[:, 0], e2_cnt_inv[:, 1], color="blue", marker=".")
                    if l2_pc_inv is not None:
                        axs[1,c].scatter(l2_pc_inv[0], l2_pc_inv[1], color='blue', marker='+')
                    if isinstance(_e2_cnt, np.ndarray) and _e2_cnt.ndim == 2:
                        axs[1,c].plot(_e2_cnt[:,0], _e2_cnt[:,1], color='red', marker=".")
                    if _l1_pc is not None:
                        axs[1,c].scatter(_l2_pc[0], _l2_pc[1], color='red', marker='+')
                        axs[1,c].set_title(f'{_p_id}:{_e_id+1}')
                        axs[1,c].axis('equal')

                fig.suptitle(f'loc ({row+1},{col+1}), p1 {p1_id}.{e1_id}, p2 {p2_id}.{e2_id}')
                plt.show()

            # if 2 first pieces have too close match, plot and confirm final choice of candidate
            if confirm:
                c_id_str = input(f'Input candidate id (0 default, -1 to skip, -2 to exit)', ).strip()
                if c_id_str.isnumeric():
                    c_id = int(c_id_str)
                # Convert default empty str to 0
                else:
                    c_id = 0

                if c_id == -1:
                    # terminate and save puzzle
                    break

                if c_id == -2:
                    # terminate and save puzzle
                    write_puzzle
                    exit()
            else:
                c_id = 0

            if 0 <= c_id < len(candidates):
                p_id = candidates[c_id]['p_id']
                e_id = candidates[c_id]['e_id'] 
            else:
                raise IndexError(f'Candidate id out of range')                        

            # Save matched piece
            piece_id = p_id + (e_id+1)/10
            print(f'Set piece {piece_id} in ({row+1},{col+1})')
            puzzle[row, col] = piece_id

    write_puzzle(puzzle)    

    return puzzle      

def read_puzzle():
    with open("puzzle.pkl", "rb") as f:
            puzzle = pickle.load(f)  
    return puzzle   

def write_puzzle(puzzle):
    with open("puzzle.pkl", "wb") as f:
        pickle.dump(puzzle, f)

    with open("puzzle.md", "w") as f:
        df = pd.DataFrame(puzzle, index=range(1, puzzle.shape[0] + 1), columns=range(1, puzzle.shape[1] + 1))
        f.write(df.to_markdown(index=True, mode='wt'))       

if __name__ == "__main__":

    # puzzle size
    M, N = 36, 28

    # tests - sets 5 - 9
    # get_pieces(p_set = 5, pieces_dir = 'row1v2', files_range = [14,27], plot_piece=True, locks_range=[2,3], select_corners=[])

    # Recognize pieces
    # get_pieces(p_set = 0, pieces_dir = 'col1')
    # get_pieces(p_set = 1, pieces_dir = 'row1v2')  
    # get_pieces(p_set = 2, pieces_dir = 'colM')
    
    # get_pieces(p_set = 10, pieces_dir = 'int20', min_locks = 4, files_range = [1,100], locks_range=[4,4])
    # get_pieces(p_set = 20, pieces_dir = 'int20', min_locks = 4, files_range = [1,100], locks_range=[4,4])
    # get_pieces(p_set = 30, pieces_dir = 'int30', min_locks = 4, files_range = [1,100], locks_range=[4,4])
    # get_pieces(p_set = 40, pieces_dir = 'int40', min_locks = 4, files_range = [1,10] , locks_range=[4,4])

    # Load pieces
    pieces = {}
    pieces = load_pieces(p_set = 0, pieces = pieces, plot_pieces=False)
    pieces = load_pieces(p_set = 1, pieces = pieces, plot_pieces=False)
    pieces = load_pieces(p_set = 2, pieces = pieces, plot_pieces=False)

    # pieces = load_pieces(p_set = 10, pieces = pieces)
    # pieces = load_pieces(p_set = 20, pieces = pieces)
    # pieces = load_pieces(p_set = 30, pieces = pieces)
    # pieces = load_pieces(p_set = 40, pieces = pieces)

    # Initiate puzzle
    # puzzle = init_puzzle(puzzle = (M, N), pieces = pieces, p_file = 'IMG_20250322_165940', p_pos = (1,N), p_edge = 4) 
    
    # Read puzzle
    puzzle = read_puzzle()
    puzzle[0, 0: N-1] = -1
    
    # Match pieces
    # puzzle = match_puzzle(puzzle, pieces, from_pos=(1,N), to_pos=(M,N) )
    # puzzle = match_puzzle(puzzle, pieces, from_pos=(1,N), to_pos=(1,1), plot_candidates=False)
    puzzle = match_puzzle(puzzle, pieces, from_pos=(1,1), to_pos=(M,1), plot_candidates=False, verbose=True)

    write_puzzle(puzzle)


