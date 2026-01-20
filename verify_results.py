
import sys

def load_csv(filename):
    with open(filename, 'r') as f:
        line = f.read().strip()
    return list(map(float, line.split(',')))

def get_val_padded(arr, c, h, w, C, H, W, P):
    # Adjust for padding
    h_in = h - P
    w_in = w - P
    
    # Boundary Check
    if h_in < 0 or h_in >= H or w_in < 0 or w_in >= W:
        return 0.0 # Zero Padding
        
    idx = int(c * (H * W) + h_in * W + w_in)
    return arr[idx]

def get_weight_val(arr, k, c, r, s, C, R, S):
    # Simple flattened access for weights [c][r][s]
    idx = int(c * (R * S) + r * S + s)
    return arr[idx]

def verify():
    # Config matching C++
    STRIDE = 2
    PADDING = 1
    DILATION = 1
    
    ia_flat = load_csv("ia.csv")
    C, H_in, W_in = 3, 10, 10
    
    filters = []
    for fname in ["fw1.csv", "fw2.csv", "fw3.csv"]:
        filters.append(load_csv(fname))
        
    K, R, S = 3, 3, 3 # Filters, Height, Width
    
    # Output Size Calculation
    # H_out = (H_in + 2*P - D*(R-1) - 1) / S + 1
    H_out = int((H_in + 2*PADDING - DILATION*(R-1) - 1) / STRIDE) + 1
    W_out = int((W_in + 2*PADDING - DILATION*(S-1) - 1) / STRIDE) + 1
    
    print(f"Expected Output Dims: {K}x{H_out}x{W_out}")
    print("Calculated Python Output:")
    
    for k in range(K):
        w_flat = filters[k]
        for y in range(H_out):
            for x in range(W_out):
                
                sum_val = 0.0
                
                # Convolution Window
                # Output(y,x) corresponds to Input(y*S, x*S) top-left corner
                y_base = y * STRIDE
                x_base = x * STRIDE
                
                for c in range(C):
                    for r in range(R):
                        for s in range(S):
                            # Effective Input Coordinate
                            h_eff = y_base + r * DILATION
                            w_eff = x_base + s * DILATION
                            
                            i_val = get_val_padded(ia_flat, c, h_eff, w_eff, C, H_in, W_in, PADDING)
                            w_val = get_weight_val(w_flat, k, c, r, s, C, R, S)
                            
                            sum_val += i_val * w_val
                            
                print(f"({k},{y},{x}): {int(sum_val)}")

if __name__ == "__main__":
    verify()
