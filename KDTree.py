import numpy as np
class KDNode:
    def __init__(self, path, point, axis, left=None, right=None):
        self.path = path
        self.point = point    # Điểm dữ liệu (ví dụ: [x, y, z])
        self.axis = axis      # Trục chia (0, 1, ..., M-1)
        self.left = left      # Nhánh trái (các điểm ≤ điểm chia)
        self.right = right    # Nhánh phải (các điểm > điểm chia)
def build_kdtree(points, depth=0):
    if not points:
        return None
    
    k = len(points[0])        # Số chiều của vector
    axis = depth % k          # Chọn trục chia theo depth
    
    # Sắp xếp và chọn median
    points = sorted(points, key = lambda x: x[1][axis])
    median = len(points) // 2
    
    path, point = points[median]
    # Xây dựng đệ quy
    return KDNode(
        path = path,
        point= point,
        axis=axis,
        left=build_kdtree(points[:median], depth + 1),
        right=build_kdtree(points[median+1:], depth + 1)
    )
def k_nearest(node, target, k, best_list=None, metric=lambda x,y: np.sqrt(np.sum((x-y)**2)) ):
    if node is None:
        return best_list
    
    if best_list is None:
        best_list = []
    
    # Tính khoảng cách và thêm vào danh sách
    current_dist = metric(node.point, target)
    best_list.append((current_dist, node.path, node.point))
    
    # Sắp xếp và giữ lại k phần tử gần nhất
    best_list.sort(key=lambda x: x[0])
    if len(best_list) > k:
        best_list = best_list[:k]
    
    # Xác định nhánh gần hơn
    if target[node.axis] <= node.point[node.axis]:
        best_list = k_nearest(node.left, target, k, best_list)
        # Kiểm tra nhánh xa nếu cần
        if (node.point[node.axis] - target[node.axis])**2 < best_list[-1][0]:
            best_list = k_nearest(node.right, target, k, best_list)
    else:
        best_list = k_nearest(node.right, target, k, best_list)
        if (target[node.axis] - node.point[node.axis])**2 < best_list[-1][0]:
            best_list = k_nearest(node.left, target, k, best_list)
    
    return best_list
