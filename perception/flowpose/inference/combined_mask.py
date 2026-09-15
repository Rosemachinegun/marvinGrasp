import numpy as np
def make_combined_mask(h, w, masks, box_ids):
    obj_ids = []
    
    if masks is None or box_ids is None:
        return None, None

    obj_ids.append([0, 0])
    combined_mask = np.zeros((h, w), dtype=np.uint8)

    # len(mask.data) = number of detected objects
    for i in range(len(masks.data)):
        mask = masks.data[i].cpu().numpy()
        obj_ids.append([i+1, box_ids[i]]) # (mask label, box id)
        gray_value = i+1
        combined_mask[mask.squeeze() > 0] = gray_value
    return combined_mask, obj_ids