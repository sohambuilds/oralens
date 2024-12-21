import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    ConfusionMatrixDisplay, 
    roc_curve, 
    auc,
    precision_recall_curve,
    average_precision_score
)
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import pickle
import cv2
from typing import Tuple, List, Dict
import pandas as pd

class ModelEvaluator:
    def __init__(
        self,
        test_dir: str,
        model_path: str,
        history_path: str,
        img_size: Tuple[int, int] = (224, 224),
        batch_size: int = 32
    ):
        self.test_dir = test_dir
        self.model_path = model_path
        self.history_path = history_path
        self.img_size = img_size
        self.batch_size = batch_size
        
        
        plt.style.use('ggplot')
        
       
        self._load_test_data()
        self._load_model()
        
    def _load_test_data(self):
        """Load and prepare test data"""
        test_datagen = ImageDataGenerator(rescale=1./255)
        self.test_generator = test_datagen.flow_from_directory(
            self.test_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        self.class_labels = list(self.test_generator.class_indices.keys())
        
    def _load_model(self):
        """Load the trained model"""
        self.model = load_model(self.model_path)
        print("Model loaded successfully from:", self.model_path)
        
    def evaluate_model(self):
        """Complete model evaluation"""
      
        self.loss, self.accuracy = self.model.evaluate(self.test_generator, verbose=1)
        print(f"\nTest Metrics:")
        print(f"Loss: {self.loss:.4f}")
        print(f"Accuracy: {self.accuracy:.4f}")

        self.Y_pred = self.model.predict(self.test_generator, verbose=1)
        self.y_pred = np.argmax(self.Y_pred, axis=1)
        self.y_true = self.test_generator.classes
      
        report = classification_report(
            self.y_true, 
            self.y_pred, 
            target_names=self.class_labels,
            output_dict=True
        )
        
  
        df_report = pd.DataFrame(report).transpose()
        print("\nClassification Report:")
        print(df_report.round(4))
        
  
        df_report.to_csv('classification_report.csv')
        
    def plot_confusion_matrix(self, normalize=True):
        """Enhanced confusion matrix visualization"""
        cm = confusion_matrix(self.y_true, self.y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='.2f' if normalize else 'd',
            cmap='Blues',
            xticklabels=self.class_labels,
            yticklabels=self.class_labels
        )
        plt.title('Normalized Confusion Matrix' if normalize else 'Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_training_history(self):
        """Enhanced training history visualization"""
        if os.path.exists(self.history_path):
            with open(self.history_path, 'rb') as f:
                history = pickle.load(f)
            
            metrics = ['accuracy', 'loss']
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            
            for idx, metric in enumerate(metrics):
                axes[idx].plot(history[metric], label=f'Training {metric}', linewidth=2)
                axes[idx].plot(history[f'val_{metric}'], label=f'Validation {metric}', linewidth=2)
                axes[idx].set_title(f'Model {metric.capitalize()}')
                axes[idx].set_xlabel('Epoch')
                axes[idx].set_ylabel(metric.capitalize())
                axes[idx].legend()
                axes[idx].grid(True)
            
            plt.tight_layout()
            plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
            plt.show()
        else:
            print("Training history not found at:", self.history_path)
            
    def visualize_predictions(self, num_samples=16):
        """Enhanced prediction visualization"""
       #random samples
        test_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
            self.test_dir,
            target_size=self.img_size,
            batch_size=num_samples,
            class_mode='categorical',
            shuffle=True
        )
        
        images, labels = next(test_generator)
        predictions = self.model.predict(images)
        

        fig = plt.figure(figsize=(20, 20))
        for i in range(num_samples):
            ax = plt.subplot(4, 4, i + 1)
            plt.imshow(images[i])
            true_label = self.class_labels[np.argmax(labels[i])]
            pred_label = self.class_labels[np.argmax(predictions[i])]
            confidence = np.max(predictions[i])
        
            color = "green" if true_label == pred_label else "red"
            plt.title(f"True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2%}",
                     color=color)
            plt.axis('off')
            
        plt.tight_layout()
        plt.savefig('prediction_samples.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def generate_gradcam_visualization(self, num_samples=5):
        """Enhanced Grad-CAM visualization"""
        def make_gradcam_heatmap(img_array, pred_index=None):
            base_mobilenet = self.model.layers[0]
            
            # Find last conv layer
            last_conv_layer = None
            for layer in reversed(base_mobilenet.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    last_conv_layer = layer
                    break
                    
            grad_model = tf.keras.models.Model(
                inputs=base_mobilenet.input,
                outputs=[last_conv_layer.output, base_mobilenet.output]
            )
            
            with tf.GradientTape() as tape:
                conv_output, predictions = grad_model(img_array)
                if pred_index is None:
                    pred_index = tf.argmax(self.model.predict(img_array)[0])
                class_channel = predictions[:, pred_index]
                
            grads = tape.gradient(class_channel, conv_output)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            conv_output = conv_output[0]
            heatmap = conv_output @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
            
            return heatmap.numpy()
            
       
        test_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
            self.test_dir,
            target_size=self.img_size,
            batch_size=num_samples,
            class_mode='categorical',
            shuffle=True
        )
        
        images, labels = next(test_generator)
        
        for idx in range(num_samples):
            img_array = np.expand_dims(images[idx], axis=0)
            heatmap = make_gradcam_heatmap(img_array)
            
            # Original image
            img = images[idx]
            
            # Process heatmap
            heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            # Superimpose heatmap on original image
            superimposed = cv2.addWeighted(
                (img * 255).astype(np.uint8),
                0.6,
                heatmap,
                0.4,
                0
            )
            
         
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            
            ax1.imshow(img)
            ax1.set_title('Original Image')
            ax1.axis('off')
            
            ax2.imshow(heatmap)
            ax2.set_title('Grad-CAM Heatmap')
            ax2.axis('off')
            
            ax3.imshow(superimposed / 255.0)
            ax3.set_title('Grad-CAM Overlay')
            ax3.axis('off')
            
            plt.tight_layout()
            plt.savefig(f'gradcam_sample_{idx}.png', dpi=300, bbox_inches='tight')
            plt.show()
            
    def run_full_evaluation(self):
        """Run complete evaluation pipeline"""
        print("Starting comprehensive model evaluation...")
        
        
        self.evaluate_model()
      
        print("\nGenerating visualizations...")
        self.plot_confusion_matrix()
        self.plot_training_history()
        self.visualize_predictions()
        self.generate_gradcam_visualization()
        
        print("\nEvaluation complete. All results have been saved.")

def main():
   
    evaluator = ModelEvaluator(
        test_dir='dataset/TEST',
        model_path='best_model.keras',
        history_path='history.pkl',
        img_size=(224, 224),
        batch_size=32
    )
    
   
    evaluator.run_full_evaluation()

if __name__ == "__main__":
    main()