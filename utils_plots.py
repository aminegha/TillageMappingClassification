import numpy as np
import matplotlib.pyplot as plt

def plot_coverage_by_pred_set_size(prediction_sets, y_test_final, args):
    """
    Calculates and prints overall coverage and coverage by prediction set size, and plots the results.
    
    Parameters:
    - prediction_sets: 2D numpy array (shape: [n_samples, n_classes]) containing the prediction sets for each sample.
    - y_test_final: List or array of true labels for each sample in the test set.
    - output_file: The filename to save the plot. Default is "coverage_by_pred_set_size.png".
    """
    # Step 1: Calculate overall coverage
    correct_predictions = np.array([prediction_sets[i, label] for i, label in enumerate(y_test_final)])
    overall_coverage = np.sum(correct_predictions) / len(y_test_final)
    print(f"Overall Coverage: {overall_coverage:.3f}")

    # Step 2: Get prediction set sizes
    pred_set_sizes = np.sum(prediction_sets, axis=1)
    
    # Step 3: Calculate coverage by prediction set size
    coverages_by_size = {}
    for size in np.unique(pred_set_sizes):
        mask = pred_set_sizes == size
        size_coverages = np.sum(correct_predictions[mask]) / np.sum(mask)  # Coverage for this set size
        coverages_by_size[size] = size_coverages
        print(f"Coverage for prediction set size {size}: {size_coverages:.3f}")
    
    # Step 4: Calculate average prediction set size
    avg_pred_set_size = np.mean(pred_set_sizes)
    print(f"Average Prediction Set Size: {avg_pred_set_size:.3f}")

    # Step 4: Plot coverage by prediction set size
    plt.figure(figsize=(10, 6))
    sizes = list(coverages_by_size.keys())
    coverages = list(coverages_by_size.values())
    
    plt.bar(sizes, coverages, color='skyblue', edgecolor='black')
    plt.xlabel("Prediction Set Size")
    plt.ylabel("Coverage")
    plt.title("Coverage by Prediction Set Size for alpha = "+ str(args.alpha))
    
    # Annotate each bar with coverage values
    for i, size in enumerate(sizes):
        plt.text(sizes[i], coverages[i] + 0.01, f'{coverages[i]*100:.2f} %', ha='center', va='bottom')
    
    # Add overall coverage and average prediction set size to the figure
    plt.text(0.5, max(coverages) * 1.05, f"Overall Coverage: {overall_coverage:.3f}", ha='center', va='bottom', fontsize=12, color='red')
    plt.text(0.5, max(coverages) * 1.10, f"Avg Set Size: {avg_pred_set_size:.3f}", ha='center', va='bottom', fontsize=12, color='green')

    # Save the figure to a file
    if args.save_cp_fig:
        plt.savefig('bioag/plots/coverage_by_pred_size/'+ str(args.model) + '_' + str(args.target_variable) + '_' + str(args.alpha) + '_' + str(args.loss) +'.png')
    plt.close()  # Close the plot to avoid displaying it in interactive environments

# Example usage:
# Assuming prediction_sets is a numpy array and y_test_final is a list or array
#, "coverage_by_pred_set_size.png")