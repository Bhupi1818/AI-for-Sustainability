from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# --- 1. Training Loop ---
num_epochs = 10 # Adjust based on your time constraints

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    # Use tqdm for a progress bar
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Train Loss: {epoch_loss:.4f}")

# --- 2. Supervised Evaluation ---
model.eval()
all_preds = []
all_labels =[]

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        
        # Get predictions
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate Metrics
acc = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average='weighted') # Weighted accounts for class imbalances

print(f"\n--- Evaluation Results ---")
print(f"Test Accuracy: {acc:.4f}")
print(f"Test F1-Score: {f1:.4f}")

# --- 3. Confusion Matrix ---
cm = confusion_matrix(all_labels, all_preds)
class_names = [id_to_category[i] for i in range(5)]

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, 
            yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title('Confusion Matrix - ResNet18 Land Cover')
plt.show()

# Brief Interpretation (To be added as a comment or print statement):
print("""
Interpretation:
Look at the diagonal of the confusion matrix to see the correctly classified instances.
Off-diagonal numbers represent misclassifications. For example, if there is a high number 
at the intersection of Actual='Vegetation' and Predicted='Cropland', it implies the model 
struggles to distinguish between farms and natural greenery, likely due to visual similarity 
in RGB channels.
""")