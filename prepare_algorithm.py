import numpy as np
import f_info
from f_info import*

def refine_cutting_plane(k, current_agent, i, dim, Na):
    m = k-1
    x = current_agent['x_memory']
    g = current_agent['g_memory']
    f = current_agent['f_memory']
    current_query = x[i*dim:(i+1)*dim, m-1]
    tilde_gjm_i = np.empty(shape=(0,0))
    tilde_fjm_i = np.empty(shape=(0,0))
    if k>2:
        tilde_fjm_i = np.zeros((Na, m))
        tilde_gjm_i = np.zeros((Na,(k-1)*dim))
        t = 0
        xim = np.empty(shape=(0,0))
        while t < k:
            xit = x[i*dim:(i+1)*dim, t]
            xim = np.vstack((xim, xit))
            for j in range(Na):
                xjm = x[i*dim:(i+1)*dim, 0:k-1]
                gjm = g[i*dim:(i+1)*dim, 0:k-1]
                fjm = f[j+1, 0:m]

                if j != i:
                    x_tem = xit - xjm
                    m = k - 1
                    sum_sol = np.zeros((1, m))
                    for p in range(m):
                        sum_sol[0, p] = gjm[:, p].T * x_tem[:, p]
                
                    f_tem = fjm + sum_sol
                    tilde_fjm_i_elem = np.max(f_tem)
                    f_idx = np.unravel_index(np.argmax(f_tem), f_tem.shape)
                    tilde_fjm_i[j, t] = tilde_fjm_i_elem
                    tilde_gjm_i_elem = gjm[:, f_idx]
                    tilde_gjm_i[j, t*dim:(t+1)*dim] = tilde_gjm_i_elem.T
                else:
                    tilde_fjm_i[j, t] = fjm[0, t]
                    tilde_gjm_i[j, t*dim:(t+1)*dim] = gjm[:, t].T
            
            t += 1
    else:
        for j in range(Na):
            xjm = x[j*dim:(j+1)*dim, 0]
            gjm = g[j*dim:(j+1)*dim, 0]
            tilde_gjm_i = np.vstack((tilde_gjm_i, gjm.T))
            fjm = gjm.T*(current_query-xjm)+f[j, 0]
            tilde_fjm_i = np.vstack((tilde_fjm_i, fjm))

        xim = current_query
    return tilde_gjm_i, tilde_fjm_i, xim