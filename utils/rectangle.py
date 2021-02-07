import cv2
import numpy as np


def is_intersect(box1, box2):
    # 判断两个矩形是否相交
    # box=(xA,yA,xB,yB)
    x01, y01, x02, y02 = box1
    x11, y11, x12, y12 = box2
 
    lx = abs((x01 + x02) / 2 - (x11 + x12) / 2)
    ly = abs((y01 + y02) / 2 - (y11 + y12) / 2)
    sax = abs(x01 - x02)
    sbx = abs(x11 - x12)
    say = abs(y01 - y02)
    sby = abs(y11 - y12)
    if lx <= (sax + sbx) / 2 and ly <= (say + sby) / 2:
        return True
    else:
        return False
    

def solve_conincide(box1, box2):
    # 计算两个矩形的重合度
    # box=(xA,yA,xB,yB)
    if not is_intersect(box1, box2):
        return 0
    
    x01, y01, x02, y02 = box1
    x11, y11, x12, y12 = box2
    col = min(x02, x12) - max(x01, x11)
    row = min(y02, y12) - max(y01, y11)
    
    intersect_area = col * row
    
    area1 = (x02 - x01) * (y02 - y01)
    area2 = (x12 - x11) * (y12 - y11)
    
    return intersect_area / area1


def merge_colse_line(lines, close=10):
    prex = lines[0]
    new_lines = [prex]

    for x in lines[1:]:
        if x - prex >= close:
            new_lines.append(x)
            prex = x

    return new_lines


def get_rects(img):
    gy_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    bi_img = cv2.adaptiveThreshold(~gy_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -10)
    h, w = bi_img.shape
    
    gap = 10
    
    # 横线
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(w//gap, 1))
    eroded = cv2.erode(bi_img, kernel, iterations = 1)
    dilatedcol = cv2.dilate(eroded, kernel, iterations = 1)
    
    # 竖线
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1,h//gap))
    eroded = cv2.erode(bi_img, kernel, iterations = 1)
    dilatedrow = cv2.dilate(eroded, kernel, iterations = 1)
    
    columns_lines = []
    rows_lines = []

    for i in range(w):
        l = np.max(dilatedrow[:,i])
        if l > 200:
            columns_lines.append(i)

    for j in range(h):
        l = np.max(dilatedcol[j,:])
        if l > 200:
            rows_lines.append(j)
            
    columns_lines = merge_colse_line(lines=columns_lines)
    rows_lines = merge_colse_line(lines=rows_lines)
    rows_lines.append(0)
    
    
    rects = []
    prev_row_points = []
    for j, y in enumerate(sorted(rows_lines)):
        row_points = []
        for i, x in enumerate(sorted(columns_lines)):
            row_points.append([x, y])

        if len(prev_row_points) > 0:
            for i_ in range(len(prev_row_points) - 1):
                j_ = i_ + 1

                # a-----b
                # |     |
                # c-----d

                x1_, y1_ = a = prev_row_points[i_]
                x2_, y2_ = b = prev_row_points[j_]
                x3_, y3_ = c = row_points[i_]
                x4_, y4_ = d = row_points[j_]

                x_, y_, w_, h_ = x1_, y1_, x2_ - x1_, y4_ - y2_
                rects.append([x_, y_, w_, h_])

        prev_row_points = row_points
    
    if len(rects) != 66:
        print('[WARNING] There are {} rectangles been detected on this invoice. The result may be wrong.'.format(len(rects)))
    
    return rects