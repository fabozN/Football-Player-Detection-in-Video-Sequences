import os
import csv
import numpy as np
import matplotlib.pyplot as plt

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

def pred_yolo_to_xyxy(raw_preds, img_w, img_h):

    boxes = []
    for xc, yc, w, h, conf in raw_preds:
        x1 = (xc - w/2) * img_w
        y1 = (yc - h/2) * img_h
        x2 = (xc + w/2) * img_w
        y2 = (yc + h/2) * img_h
        boxes.append([x1, y1, x2, y2, conf])

    return boxes

def match_prediction(pred, gt_boxes, matched_gt, iou_thresh=0.5):

    best_iou = 0
    best_gt_idx = -1

    for i, gt in enumerate(gt_boxes):
        iou_val = iou(pred, gt)
        if iou_val > best_iou:
            best_iou = iou_val
            best_gt_idx = i

    if best_iou >= iou_thresh:
        if best_gt_idx not in matched_gt:
            matched_gt.add(best_gt_idx)
            return 1, 0
        else:
            return 0, 1 
    else:
        return 0, 1    


def evaluate_method_yolo(pred_dir, gt_dir, img_size=(1920,1080)):

    all_predictions = []  
    total_gt = 0

    for fname in sorted(os.listdir(gt_dir)):
        if not fname.endswith(".txt"):
            continue

        gt_path = os.path.join(gt_dir, fname)
        pred_path = os.path.join(pred_dir, fname.replace(".txt", "_bboxes.npy"))

        gt_boxes = yolo_to_xyxy(gt_path, *img_size)
        total_gt += len(gt_boxes)

        preds = []
        if os.path.exists(pred_path):
            raw = np.load(pred_path, allow_pickle=True)
            preds = pred_yolo_to_xyxy(raw, *img_size)

        preds.sort(key=lambda x: -x[4])

        matched_gt = set()

        print(f"\nFrame: {fname}")
        print(f"GT: {len(gt_boxes)}  Pred: {len(preds)}")

        for i, (x1, y1, x2, y2, conf) in enumerate(preds):
            pred_box = [x1, y1, x2, y2]

            best_iou = 0.0
            for gt in gt_boxes:
                best_iou = max(best_iou, iou(pred_box, gt))
            print(f"Pred {i}: conf={conf:.2f}, best IoU={best_iou:.4f}")

            TP, FP = match_prediction(pred_box, gt_boxes, matched_gt)
            all_predictions.append([conf, TP, FP])

    all_predictions.sort(key=lambda x: -x[0])

    cum_TP = 0
    cum_FP = 0
    precisions = []
    recalls = []

    for conf, TP, FP in all_predictions:
        cum_TP += TP
        cum_FP += FP
        precision = cum_TP / (cum_TP + cum_FP + 1e-9)
        recall = cum_TP / (total_gt + 1e-9)
        precisions.append(precision)
        recalls.append(recall)

    ap = 0.0
    for i in range(1, len(recalls)):
        ap += precisions[i] * (recalls[i] - recalls[i-1])

    precision_final = precisions[-1]
    recall_final = recalls[-1]

    plt.figure(figsize=(6,6))
    plt.plot(recalls, precisions, marker='.')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (YOLO ft)")
    plt.grid(True)
    plt.savefig("results/finetuned/sequence_move/PR_curve.png", dpi=200)
    plt.close()
    print("Grafico PR salvato come pr_curve_yolo.png")

    with open("results/finetuned/sequence/sequence.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["precision", "recall", "AP"])
        writer.writerow([precision_final, recall_final, ap])

    return precision_final, recall_final, ap


if __name__ == "__main__":

    precision, recall, ap = evaluate_method_yolo(
        pred_dir="data/detections/finetuned_test/sequence_move",
        gt_dir="dataset_combined/A_test/labels"
    )

    print("Precision:", precision)
    print("Recall:", recall)
    print("AP:", ap)
