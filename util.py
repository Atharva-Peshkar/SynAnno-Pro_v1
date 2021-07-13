import os
import numpy as np

def mkdir(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

def writetxt(filename, content):
    a= open(filename,'w')
    if isinstance(content, (list,)):
        for ll in content:
            a.write(ll)
            if '\n' not in ll:
                a.write('\n')
    else:
        a.write(content)
    a.close()

def readtxt(filename):
    a= open(filename)
    content = a.readlines()
    a.close()
    return content

def rotateIm(image, angle, center=None, scale=1.0):
    if angle == 0:
        return image
    else:
        import cv2
        # grab the dimensions of the image
        (h, w) = image.shape[:2]
        # if the center is None, initialize it as the center of
        # the image
        if center is None:
            center = (w // 2, h // 2)
        # perform the rotation
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(image, M, (w, h))
        return rotated

def bfly(bfly_db, x0, x1, y0, y1, z0, z1, tile_sz, dt=np.uint8, tile_st=[0,0], tile_ratio=1, resize_order=1, ndim=1,black=128):
    # x: column
    # y: row
    # no padding at the boundary
    # st: starting index 0 or 1
    result = np.zeros((z1-z0, y1-y0, x1-x0, ndim), dt)
    c0 = x0 // tile_sz # floor
    c1 = (x1 + tile_sz-1) // tile_sz # ceil
    r0 = y0 // tile_sz
    r1 = (y1 + tile_sz-1) // tile_sz
    for z in range(z0, z1):
        pattern = bfly_db[z]
        for row in range(r0, r1):
            for column in range(c0, c1):
                path = pattern.format(row=row+tile_st[0], column=column+tile_st[1])
                if not os.path.exists(path): 
                    #return None
                    if dt==np.uint8:
                        patch = black*np.ones((tile_sz,tile_sz,ndim),dtype=np.uint8)
                    else:
                        patch = black*np.ones((tile_sz,tile_sz,3),dtype=np.uint8)
                else:
                    if path[-3:]=='tif':
                        import tifffile
                        patch = tifffile.imread(path)
                    else:
                        from imageio import imread
                        #from scipy.misc import imread
                        patch = imread(path)
                if tile_ratio != 1:
                    # scipy.misc.imresize: only do uint8
                    from scipy.ndimage import zoom
                    patch = zoom(patch, tile_ratio, order=resize_order)
                if patch.ndim==2:
                    patch=patch[:,:,None]
                
                # last tile may not be full
                xp0 = column * tile_sz
                xp1 = xp0 + patch.shape[1]
                #xp1 = (column+1) * tile_sz
                yp0 = row * tile_sz
                yp1 = yp0 + patch.shape[0]
                #yp1 = (row + 1) * tile_sz
                if patch is not None:
                    x0a = max(x0, xp0)
                    x1a = min(x1, xp1)
                    y0a = max(y0, yp0)
                    y1a = min(y1, yp1)
                    sz = result[z-z0, y0a-y0:y1a-y0, x0a-x0:x1a-x0].shape
                    if resize_order==0 or dt!=np.uint8:
                        if dt==np.uint8:
                            result[z-z0, y0a-y0:y1a-y0, x0a-x0:x1a-x0] = patch[y0a-yp0:y1a-yp0, x0a-xp0:x1a-xp0,0].reshape(sz)
                        else:
                            result[z-z0, y0a-y0:y1a-y0, x0a-x0:x1a-x0] = vast2Seg(patch[y0a-yp0:y1a-yp0, x0a-xp0:x1a-xp0]).reshape(sz)
                    else:
                        result[z-z0, y0a-y0:y1a-y0, x0a-x0:x1a-x0] = patch[y0a-yp0:y1a-yp0, x0a-xp0:x1a-xp0]
    return result

