
params = ['Results for', 'Accuracy:', 'weighted avg']
cm_start = False
cm_count = 0
with open('nids_results.txt', 'r') as f:
    for line in f:
        if any(p in line for p in params):
            print(line.strip())
        if 'Confusion Matrix:' in line:
            print(line.strip())
            cm_start = True
            cm_count = 0
            continue
        if cm_start:
            if line.strip().startswith('['):
                print(line.strip())
                cm_count += 1
            if cm_count >= 4: # Assuming 4 classes
                cm_start = False
