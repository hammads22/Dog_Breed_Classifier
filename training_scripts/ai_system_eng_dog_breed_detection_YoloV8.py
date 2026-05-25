# -*- coding: utf-8 -*-
"""AI System Eng - Dog Breed Detection"""

import os
import shutil
from collections import Counter, defaultdict
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from roboflow import Roboflow

# ==========================================
# FUNCTION DEFINITIONS
# ==========================================
def count_class_distribution(label_path, class_names):
    class_counts = Counter()
    label_files = sorted([f for f in os.listdir(label_path) if f.endswith('.txt')])

    for label_file in label_files:
        with open(os.path.join(label_path, label_file), 'r') as f:
            annotations = f.readlines()
            for annot in annotations:
                class_idx = int(annot.split()[0])
                class_counts[class_idx] += 1

    sorted_class_counts = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    sorted_class_names = [class_names[i] for i, _ in sorted_class_counts]
    sorted_counts = [count for _, count in sorted_class_counts]

    return sorted_class_names, sorted_counts

def display_one_image_per_class_with_bboxes(image_dir, label_dir, class_names):
    images_displayed = {class_name: False for class_name in class_names}
    for filename in os.listdir(image_dir):
        if filename.endswith(('.jpg', '.png')):
            image_path = os.path.join(image_dir, filename)
            label_path = os.path.join(label_dir, filename[:-4] + '.txt')

            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    img = cv2.imread(image_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    height, width, _ = img.shape

                    for line in f:
                        class_index, x_center, y_center, w, h = map(float, line.split())
                        class_name = class_names[int(class_index)]

                        if not images_displayed[class_name]:
                            x1 = int((x_center - w / 2) * width)
                            y1 = int((y_center - h / 2) * height)
                            x2 = int((x_center + w / 2) * width)
                            y2 = int((y_center + h / 2) * height)

                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(img, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                            print(f"\n!!! ATTENTION: Opening Image for {class_name}. CLOSE THE WINDOW TO CONTINUE !!!")
                            plt.imshow(img)
                            plt.title(f"Class: {class_name}")
                            plt.axis('off')
                            plt.show()
                            images_displayed[class_name] = True
                            break

def bounding_box_statistics(copy_label_dir):
    bbox_stats = defaultdict(list)
    label_files = sorted([f for f in os.listdir(copy_label_dir) if f.endswith('.txt')])

    for label_file in label_files:
        label_path = os.path.join(copy_label_dir, label_file)
        with open(label_path, 'r') as f:
            labels = f.readlines()

        for label in labels:
            class_idx, x_center, y_center, width, height = map(float, label.strip().split())
            bbox_area = width * height
            bbox_stats[class_idx].append(bbox_area)

    stats_summary = {
        class_idx: {
            'Count': len(areas),
            'Mean': sum(areas) / len(areas) if areas else 0,
            'Min': min(areas) if areas else 0,
            'Max': max(areas) if areas else 0
        }
        for class_idx, areas in sorted(bbox_stats.items())
    }

    stats_df = pd.DataFrame(stats_summary).T
    stats_df.index.name = 'Class'
    return stats_df

def validate_dataset_coordinates(labels_path):
    invalid_files = []
    for label_file in os.listdir(labels_path):
        with open(os.path.join(labels_path, label_file), 'r') as f:
            for line in f:
                class_idx, x_center, y_center, width, height = map(float, line.strip().split())
                if not all(0 < coord <= 1 for coord in [x_center, y_center, width, height]):
                    invalid_files.append(label_file)
                    break
    return invalid_files

def clean_bounding_boxes_on_copy(copy_image_dir, copy_label_dir, min_threshold=0.001, max_threshold=0.9):
    cleaned_labels = 0
    total_labels = 0
    image_files = sorted(os.listdir(copy_image_dir))
    label_files = sorted(os.listdir(copy_label_dir))

    for image_file, label_file in zip(image_files, label_files):
        image_path = os.path.join(copy_image_dir, image_file)
        label_path = os.path.join(copy_label_dir, label_file)
        image_base = os.path.splitext(image_file)[0]
        label_base = os.path.splitext(label_file)[0]

        if image_base != label_base:
            continue

        with open(label_path, 'r') as f:
            labels = f.readlines()

        new_labels = []
        for label in labels:
            total_labels += 1
            class_idx, x_center, y_center, width, height = map(float, label.strip().split())
            bbox_area = width * height

            if min_threshold <= bbox_area <= max_threshold:
                new_labels.append(label)
            else:
                cleaned_labels += 1

        if len(new_labels) == 0:
            os.remove(image_path)
            os.remove(label_path)
        else:
            with open(label_path, 'w') as f:
                for new_label in new_labels:
                    f.write(new_label)

    print(f"Total boxes processed: {total_labels} | Boxes removed: {cleaned_labels}")

def limit_images_per_class(image_dir, label_dir, max_images_per_class=250):
    class_counts = defaultdict(int)
    removed_count = 0
    for filename in os.listdir(image_dir):
        if filename.endswith(('.jpg', '.png')):
            image_path = os.path.join(image_dir, filename)
            label_path = os.path.join(label_dir, filename[:-4] + '.txt')

            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                should_delete = False
                for line in lines:
                    class_idx = int(line.split()[0])
                    class_counts[class_idx] += 1
                    if class_counts[class_idx] > max_images_per_class:
                        should_delete = True
                        break
                
                if should_delete:
                    os.remove(image_path)
                    os.remove(label_path)
                    removed_count += 1
                    
    print(f"Total images removed to enforce limit: {removed_count}")


# ==========================================
# MAIN EXECUTION BLOCK (Crucial for Windows!)
# ==========================================
if __name__ == '__main__':
    print("\n--- [CHECKPOINT 1] Starting Script & Loading Roboflow... ---")
    rf = Roboflow(api_key="eGsEbW5nGpFjTU1Ml5Th")
    project = rf.workspace("igor-romanica-gmail-com").project("stanford-dogs-0pff9")
    version = project.version(3)
    dataset = version.download("yolov8")

    print("\n--- [CHECKPOINT 2] Dataset Downloaded. Reading YAML... ---")
    image_dir = 'Stanford-dogs-3/train/images'
    label_dir = 'Stanford-dogs-3/train/labels'

    with open('Stanford-dogs-3/data.yaml', 'r') as f:
        data_yaml = yaml.safe_load(f)

    class_names = data_yaml['names']

    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')])
    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.txt')])

    print(f"Number of images: {len(image_files)}")
    print(f"Number of labels: {len(label_files)}")
    print(f"Number of classes: {len(class_names)}")

    print("\n--- [CHECKPOINT 3] Calculating Class Distribution... ---")
    label_path = 'Stanford-dogs-3/train/labels'
    sorted_class_names, sorted_counts = count_class_distribution(label_path, class_names)

    print("\n!!! ATTENTION: Opening Class Distribution Plot. CLOSE THE WINDOW TO CONTINUE SCRIPT !!!")
    plt.figure(figsize=(10, 6))
    plt.bar(sorted_class_names, sorted_counts, color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.title('Sorted Class Distribution')
    plt.xlabel('Class Label')
    plt.ylabel('Frequency')
    plt.show()

    print("\n--- [CHECKPOINT 4] Displaying Sample Bounding Boxes... ---")
    #display_one_image_per_class_with_bboxes(image_dir, label_dir, class_names)

    print("\n--- [CHECKPOINT 5] Calculating Bounding Box Statistics... ---")
    stats = bounding_box_statistics(label_dir)
    
    bbox_area = []
    for label_file in label_files:
        with open(os.path.join(label_dir, label_file), 'r') as f:
            labels = f.readlines()
            for label in labels:
                parts = label.strip().split()
                if len(parts) == 5:
                    width = float(parts[3])
                    height = float(parts[4])
                    bbox_area.append(width * height)

    print(f"Bounding boxes number: {len(bbox_area)}")
    print("\n!!! ATTENTION: Opening BBox Area Histogram. CLOSE THE WINDOW TO CONTINUE !!!")
    plt.figure(figsize=(10, 6))
    plt.hist(bbox_area, bins=50, color='skyblue')
    plt.title("Distribution of Bounding Box Areas")
    plt.xlabel("Area")
    plt.ylabel("Frequency")
    plt.show()

    print("\n--- [CHECKPOINT 6] Starting Data Cleaning & Folder Cloning... ---")
    base_path = 'Stanford-dogs-3'
    folders_to_clone = ['train', 'test', 'valid']

    for folder in folders_to_clone:
        original_folder = os.path.join(base_path, folder)
        cloned_folder = os.path.join(base_path, f"{folder}_model")

        if not os.path.exists(cloned_folder):
            os.makedirs(cloned_folder)

        if os.path.exists(original_folder):
            for item in os.listdir(original_folder):
                source = os.path.join(original_folder, item)
                destination = os.path.join(cloned_folder, item)
                if os.path.isdir(source):
                    if not os.path.exists(destination):
                        shutil.copytree(source, destination)
                else:
                    shutil.copy2(source, destination)

    print("Cloning complete: created train_model, test_model, and valid_model folders.")

    print("\n--- [CHECKPOINT 7] Validating Coordinates... ---")
    model_image_dir = 'Stanford-dogs-3/train_model/images'
    model_label_dir = 'Stanford-dogs-3/train_model/labels'
    invalid_labels = validate_dataset_coordinates(model_label_dir)
    if invalid_labels:
        print(f"Found {len(invalid_labels)} files with invalid coordinates.")
    else:
        print("All coordinates in the original dataset are valid.")

    print("\n--- [CHECKPOINT 8] Cleaning Bounding Boxes on Copy... ---")
    clean_bounding_boxes_on_copy(model_image_dir, model_label_dir, min_threshold=0.001, max_threshold=0.9)

    print("\n--- [CHECKPOINT 9] Limiting Images Per Class... ---")
    limit_images_per_class(model_image_dir, model_label_dir, max_images_per_class=250)

    print("\n--- [CHECKPOINT 10] Preparing Folders for YOLO Training... ---")
    dataset_path = 'Stanford-dogs-3'
    train_old_path = os.path.join(dataset_path, 'train_old')
    train_model_path = os.path.join(dataset_path, 'train_model')
    train_new_path = os.path.join(dataset_path, 'train')

    # Fix for WinError 183: Remove train_old if it exists from a previous run
    if os.path.exists(train_old_path) and os.path.exists(train_new_path):
        print("Cleaning up old backup folder...")
        shutil.rmtree(train_old_path)

    if os.path.exists(train_new_path):
        os.rename(train_new_path, train_old_path)
    if os.path.exists(train_model_path):
        os.rename(train_model_path, train_new_path)
    print("Folder paths successfully swapped.")

    print("\n--- [CHECKPOINT 11] INIT YOLOv8 TRAINING... ---")
    from ultralytics import YOLO
    model = YOLO('yolov8n.pt')

    print("Starting training for 20 epochs...")
    # Because we are protected by __main__, the workers spawned here won't crash!
    results = model.train(data='Stanford-dogs-3/data.yaml',
                          epochs=20,
                          imgsz=608,
                          batch=16,
                          augment=True)

    model_path = 'Stanford-dogs-3/20_epoch_model.pt'
    model.save(model_path)
    print(f"\n--- [CHECKPOINT 12] MODEL SAVED at {model_path} ---")

    print("\n--- [CHECKPOINT 13] Running Predictions on Test Set... ---")
    test_images_path = 'Stanford-dogs-3/test/images'
    test_image_files = sorted([f for f in os.listdir(test_images_path) if f.endswith('.jpg') or f.endswith('.png')])[:10]

    for image_file in test_image_files:
        image_path = os.path.join(test_images_path, image_file)
        reslist = model.predict(image_path)

        for res in reslist[:10]:
            img = res.orig_img
            for box in res.boxes:
                x1, y1, x2, y2 = box.xyxy.cpu().numpy().astype(int)[0]
                cls = int(box.cls[0].cpu().numpy())
                conf = box.conf[0].cpu().numpy()
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                label = f"{model.names[cls]}: {conf:.2f}"
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
            print(f"\n!!! ATTENTION: Opening Prediction for {image_file}. CLOSE THE WINDOW TO CONTINUE !!!")
            plt.figure(figsize=(10, 10))
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.title(image_file)
            plt.show()

    print("\n--- [CHECKPOINT 14] Starting Model Validation... ---")
    val_res = model.val(data='Stanford-dogs-3/data.yaml')

    print("\n!!! ATTENTION: Opening Confusion Matrix. CLOSE THE WINDOW TO FINISH SCRIPT !!!")
    conf_matrix = val_res.confusion_matrix.matrix
    import seaborn as sns
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt=".1f", cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

    print("\n--- [CHECKPOINT 15] Calculating Final Metrics... ---")
    precisions, recalls, f1_scores = [], [], []
    for i in range(len(class_names)):
        tp = conf_matrix[i, i]
        fp = conf_matrix[:, i].sum() - tp
        fn = conf_matrix[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    results_df = pd.DataFrame({'Class': class_names, 'Precision': precisions, 'Recall': recalls, 'F1-Score': f1_scores})
    results_df.set_index('Class', inplace=True)
    print(results_df)

    print("\n--- [SCRIPT COMPLETE] ---")