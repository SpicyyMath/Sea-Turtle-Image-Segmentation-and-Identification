# evaluation.py

import os
import argparse
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from sklearn.metrics import jaccard_score, classification_report
import joblib
from tqdm import tqdm

# =================================================================================
# 1. Data Loader (specifically for Deep Learning Segmentation Models)
#    - This part is extracted and optimized from your deep_learning.ipynb to ensure
#      data handling is consistent between training and evaluation.
# =================================================================================
class SeaTurtleTestDataset(Dataset):
    """
    Custom Dataset class for loading sea turtle test images and masks.
    """
    def __init__(self, data_path, transform=None):
        self.image_paths = sorted([os.path.join(data_path, 'images', f) for f in os.listdir(os.path.join(data_path, 'images')) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        self.mask_paths = sorted([os.path.join(data_path, 'masks', f) for f in os.listdir(os.path.join(data_path, 'masks')) if f.lower().endswith('.png')])
        self.transform = transform if transform else A.Compose([A.Resize(256, 256), ToTensorV2()])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load the corresponding multi-channel mask
        image_id = os.path.basename(image_path).split('_')[0]
        matching_mask_paths = sorted([p for p in self.mask_paths if os.path.basename(p).startswith(image_id) and '_channel_' in p])
        
        # Exclude the background channel (_channel_0.png)
        matching_mask_paths = [p for p in matching_mask_paths if not p.endswith('_channel_0.png')]
        
        if not matching_mask_paths:
            # If no corresponding mask is found, create and return an empty mask
            mask = np.zeros((256, 256), dtype=np.uint8)
        else:
            mask = self._load_multichannel_mask(matching_mask_paths)

        # Apply augmentations/transformations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask

    def _load_multichannel_mask(self, mask_paths):
        first_mask = cv2.imread(mask_paths[0], cv2.IMREAD_GRAYSCALE)
        mask = np.zeros(first_mask.shape, dtype=np.uint8)
        # Assign class indices based on the channel number (e.g., _channel_1.png)
        for path in mask_paths:
            channel_mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            class_index = int(os.path.basename(path).split('_channel_')[-1].split('.')[0])
            mask[channel_mask > 0] = class_index
        return mask

# =================================================================================
# 2. Feature Loader (specifically for Traditional Machine Learning Classification Models)
#    - This function loads the features you previously extracted and saved using HOG or SIFT.
# =================================================================================
def load_features_for_classification(features_path):
    """
    Loads pre-extracted features and labels from folders.
    Assumes features are stored in 'head', 'limbs', and 'shell' subdirectories.
    """
    X, y = [], []
    # Mapping from folder names to class labels
    label_map = {'shell': 'Shell', 'head': 'Head', 'limbs': 'Limbs'}
    
    for part_folder, label in label_map.items():
        part_path = os.path.join(features_path, part_folder)
        if not os.path.exists(part_path):
            print(f"Warning: Feature folder {part_path} not found, skipping.")
            continue
            
        for filename in os.listdir(part_path):
            if filename.endswith('.npy'):
                feature = np.load(os.path.join(part_path, filename))
                # SIFT features might be (N, 128), so we take the mean to create a single feature vector
                if feature.ndim > 1:
                    feature = np.mean(feature, axis=0)
                X.append(feature)
                y.append(label)
    
    return np.array(X), np.array(y)


# =================================================================================
# 3. Evaluation Logic
# =================================================================================
def evaluate_segmentation_model(model, loader, device, num_classes):
    """Function to evaluate the deep learning segmentation model."""
    model.eval()
    all_preds = []
    all_masks = []
    
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Evaluating Segmentation Model"):
            images = images.to(device)
            masks = masks.long().numpy() # Ground truth labels
            
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy() # Predicted labels
            
            all_preds.append(preds.flatten())
            all_masks.append(masks.flatten())

    all_preds = np.concatenate(all_preds)
    all_masks = np.concatenate(all_masks)

    # Calculate IoU (Jaccard Score) for each class, ignoring the background class (index 0)
    class_labels = ["Background", "Shell", "Limbs", "Head"] # Corresponds to indices 0, 1, 2, 3
    ious = {}
    for i in range(1, num_classes):
        iou = jaccard_score(all_masks == i, all_preds == i)
        ious[class_labels[i]] = iou
        print(f"IoU for class '{class_labels[i]}': {iou:.4f}")
        
    mean_iou = np.mean(list(ious.values()))
    print("-" * 30)
    print(f"Mean IoU: {mean_iou:.4f}")

def evaluate_classification_model(model, X_test, y_test):
    """Function to evaluate the traditional machine learning classification model."""
    print("--- Evaluating Classification Model ---")
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions)
    print("Classification Report:\n", report)

# =================================================================================
# 4. Main Program Entry Point
# =================================================================================
def main():
    parser = argparse.ArgumentParser(description="Evaluation script for the Sea Turtle Image Segmentation and Identification project.")
    
    parser.add_argument('--model-path', type=str, required=True, help="Path to the trained model file (.pth or .pkl)")
    parser.add_argument('--data-path', type=str, required=True, help="Root directory of the preprocessed data")
    parser.add_argument('--model-type', type=str, required=True, choices=['unet', 'deeplabv3', 'pspnet', 'svm', 'rf'], help="Type of model to evaluate")
    parser.add_argument('--encoder', type=str, default='resnet34', help="Encoder (backbone) for the deep learning model")
    parser.add_argument('--num-classes', type=int, default=4, help="Number of classes for the segmentation task (including background)")
    parser.add_argument('--batch-size', type=int, default=8, help="Batch size for evaluation")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Select evaluation flow based on model type ---
    if args.model_type in ['unet', 'deeplabv3', 'pspnet']:
        # --- Deep Learning Segmentation Model Evaluation ---
        print(f"Loading deep learning model: {args.model_type.upper()} with {args.encoder} backbone")
        
        # 1. Initialize model architecture
        if args.model_type == 'unet':
            model = smp.Unet(encoder_name=args.encoder, encoder_weights='imagenet', in_channels=3, classes=args.num_classes)
        elif args.model_type == 'deeplabv3':
            model = smp.DeepLabV3Plus(encoder_name=args.encoder, encoder_weights='imagenet', in_channels=3, classes=args.num_classes)
        elif args.model_type == 'pspnet':
            model = smp.PSPNet(encoder_name=args.encoder, encoder_weights='imagenet', in_channels=3, classes=args.num_classes)
        
        # 2. Load trained weights
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.to(device)
        
        # 3. Prepare the test data loader
        test_data_path = os.path.join(args.data_path, 'test')
        test_transform = A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
        test_dataset = SeaTurtleTestDataset(test_data_path, transform=test_transform)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        # 4. Run evaluation
        evaluate_segmentation_model(model, test_loader, device, args.num_classes)

    elif args.model_type in ['svm', 'rf']:
        # --- Traditional Machine Learning Classification Model Evaluation ---
        print(f"Loading traditional machine learning model: {args.model_type.upper()}")
        
        # 1. Load the model
        model = joblib.load(args.model_path)
        
        # 2. Load pre-computed features
        # Assuming feature folders are 'features_sift' or 'features_hog' under data_path
        features_path = os.path.join(args.data_path, 'features_sift') # Using SIFT by default, you can change this
        X_test, y_test = load_features_for_classification(features_path)
        
        if X_test.size == 0:
            print(f"Error: No feature files found in path {features_path}. Please check your data path and structure.")
            return
            
        # 3. Run evaluation
        evaluate_classification_model(model, X_test, y_test)

if __name__ == '__main__':
    print("--- Project Evaluation Script ---")
    print("Example Usage:")
    print("1. To evaluate a U-Net segmentation model:")
    print("   python evaluation.py --model-path best_unet_seaturtle.pth --data-path ./preprocessed_data --model-type unet")
    print("\n2. To evaluate an SVM classification model (assuming features are in 'preprocessed_data/features_sift'):")
    print("   python evaluation.py --model-path hog_svm_classifier.pkl --data-path ./preprocessed_data --model-type svm")
    print("-" * 50)
    main()