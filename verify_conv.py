
# Pure Python implementation
ia_str = "1,0,2,0,3,4,0,5,6,0,0,2,0,3,0,4,5,0,6,7,2,0,3,0,4,0,6,7,0,8,0,3,0,4,5,0,7,0,8,9,3,0,4,5,0,6,0,8,9,0,0,4,5,0,6,0,8,9,0,10,4,5,0,6,0,7,8,0,10,11,5,0,6,0,7,8,0,10,11,0,0,6,0,7,8,0,10,0,11,12,6,0,7,8,0,10,0,11,12,0,0,1,0,2,0,3,4,0,5,6,1,0,2,0,3,0,4,5,0,6,0,2,0,3,0,4,0,6,7,0,2,0,3,0,4,5,0,7,0,8,0,3,0,4,5,0,6,0,8,9,3,0,4,5,0,6,0,8,9,0,0,4,5,0,6,0,7,8,0,10,4,5,0,6,0,7,8,0,10,11,5,0,6,0,7,8,0,10,0,11,0,6,0,7,8,0,10,0,11,12,2,1,0,3,2,0,4,3,0,1,1,0,2,1,0,3,2,0,4,3,0,2,1,0,3,2,0,4,3,0,3,1,0,2,1,0,4,3,0,5,2,0,3,1,0,4,2,0,5,3,0,3,1,0,4,2,0,5,3,0,4,2,0,3,1,0,5,3,0,6,3,0,4,2,0,5,3,0,6,4,0,4,2,0,5,3,0,6,4,0,5,3,0,4,2,0,6,4,0,7"

def parse_csv(s):
    return [float(x) for x in s.split(',')]

ia_flat = parse_csv(ia_str)
# 3x10x10
IA = []
idx = 0
for c in range(3):
    chan = []
    for h in range(10):
        row = []
        for w in range(10):
            row.append(ia_flat[idx])
            idx += 1
        chan.append(row)
    IA.append(chan)

fw1_str = "1,0,-1,0,1,0,-1,0,1,0,1,0,1,0,-1,0,-1,0,1,0,0,0,-1,0,0,0,1"
fw1_flat = parse_csv(fw1_str)
FW1 = []
idx = 0
for c in range(3): # 3 channels
    chan = []
    for r in range(3):
        row = []
        for s in range(3):
            row.append(fw1_flat[idx])
            idx+=1
        chan.append(row)
    FW1.append(chan)

fw2_str = "0,1,0,1,0,-1,0,-1,0,1,0,1,0,-1,0,1,0,1,0,1,0,1,0,1,0,1,0"
fw2_flat = parse_csv(fw2_str)
FW2 = []
idx=0
for c in range(3):
    chan = []
    for r in range(3):
        row = []
        for s in range(3):
            row.append(fw2_flat[idx])
            idx+=1
        chan.append(row)
    FW2.append(chan)

fw3_str = "-1,0,1,0,2,0,1,0,-1,1,0,0,0,1,0,0,0,1,0,-1,0,-1,0,1,0,1,0"
fw3_flat = parse_csv(fw3_str)
FW3 = []
idx=0
for c in range(3):
    chan = []
    for r in range(3):
        row = []
        for s in range(3):
            row.append(fw3_flat[idx])
            idx+=1
        chan.append(row)
    FW3.append(chan)

FWs = [FW1, FW2, FW3]

def get_input_val(c, h, w):
    if h < 0 or h >= 10 or w < 0 or w >= 10:
        return 0.0
    return IA[c][h][w]

# Output 3x8x8
OA = [[[0.0 for _ in range(8)] for _ in range(8)] for _ in range(3)]

for k in range(3):
    for c in range(3):
        # 3x3 conv
        kernel = FWs[k][c]
        for h_out in range(8):
            for w_out in range(8):
                # Stride 1, Pad 1, Dilation 2
                # y_in = y_out * U + y_w * D - P
                # y_in = h_out * 1 + r * 2 - 1
                val = 0.0
                for r in range(3):
                    for s in range(3):
                        h_in = h_out + r * 2 - 1
                        w_in = w_out + s * 2 - 1
                        val += get_input_val(c, h_in, w_in) * kernel[r][s]
                OA[k][h_out][w_out] += val

print("Values:")
for k in range(3):
    for h in range(8):
        for w in range(8):
             val = OA[k][h][w]
             if abs(val) > 1e-5:
                 print(f"({k},{h},{w}): {int(val)}")

