import os
import csv
import numpy as np

def iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter_area

    return inter_area / union if union > 0 else 0

def yolo_to_xyxy(label_path, img_w, img_h):

    boxes = []
    with open(label_path, "r") as f:
        for line in f.readlines():
            cls, xc, yc, w, h = map(float, line.strip().split())
            x1 = (xc - w/2) * img_w
            y1 = (yc - h/2) * img_h
            x2 = (xc + w/2) * img_w
            y2 = (yc + h/2) * img_h
            boxes.append([x1, y1, x2, y2])

    return boxes


def evaluate_frame(pred_boxes, gt_boxes, iou_thresh=0.5):

    matched_gt = set()
    TP = 0
    FP = 0

    for pred in pred_boxes:
        best_iou = 0
        best_gt_idx = -1

        for i, gt in enumerate(gt_boxes):
            iou_val = iou(pred, gt)
            if iou_val > best_iou:
                best_iou = iou_val
                best_gt_idx = i

        if best_iou >= iou_thresh:
            if best_gt_idx not in matched_gt:
                TP += 1
                matched_gt.add(best_gt_idx)
            else:
                FP += 1
        else:
            FP += 1

    FN = len(gt_boxes) - len(matched_gt)
    
    return TP, FP, FN

def evaluate_method_sp(pred_dir, gt_dir, img_size=(1920,1080)):

    total_TP = 0
    total_FP = 0
    total_FN = 0

    for fname in sorted(os.listdir(gt_dir)):
        if not fname.endswith(".txt"):
            continue

        gt_path = os.path.join(gt_dir, fname)
        pred_path = os.path.join(pred_dir, fname.replace(".txt", "_bboxes.npy"))

        gt_boxes = yolo_to_xyxy(gt_path, *img_size)

        pred_boxes_xyxy = []
        if os.path.exists(pred_path):
            raw = np.load(pred_path, allow_pickle=True)
            for (x, y, w, h) in raw:
                pred_boxes_xyxy.append([x, y, x+w, y+h])
        
        print(f"\nFrame: {fname}")
        print(f"GT: {len(gt_boxes)}  Pred: {len(pred_boxes_xyxy)}")

        if len(pred_boxes_xyxy) > 0:
            for i, pred in enumerate(pred_boxes_xyxy):
                best_iou = 0.0
                for gt in gt_boxes:
                    best_iou = max(best_iou, iou(pred, gt))
                print(f"Pred {i}: best IoU = {best_iou:.4f}")

        TP, FP, FN = evaluate_frame(pred_boxes_xyxy, gt_boxes)
        total_TP += TP
        total_FP += FP
        total_FN += FN

    precision = total_TP / (total_TP + total_FP + 1e-9)
    recall = total_TP / (total_TP + total_FN + 1e-9)
    ap = precision * recall 

    with open("results/mog2_solo_mask/sequence/sequence.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["precision", "recall", "AP"])
        writer.writerow([precision, recall, ap])

    return precision, recall, ap


if __name__ == "__main__":

    precision, recall, ap = evaluate_method_sp(
        pred_dir="data/detections/mog2_solo_mask/test/sequence",
        gt_dir="dataset_combined/B_test/labels"
    )

    print("Precision:", precision)
    print("Recall:", recall)
    print("AP:", ap)


