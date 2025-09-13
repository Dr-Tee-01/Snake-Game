import matplotlib.pyplot as plt
from IPython import display

# Enable interactive plotting (updates without blocking execution)
plt.ion()

def plot(scores, mean_scores):
    """
    Plots the training progress of the Snake AI.
    
    Args:
        scores (list): List of game scores achieved.
        mean_scores (list): List of average scores over time.
    """
    # Clear previous output (useful for Jupyter/IPython)
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()   # clear current figure

    # Set plot title and labels
    plt.title("Snake AI Training Progress")
    plt.xlabel("Number of Games")
    plt.ylabel("Score")

    # Plot the raw scores and their moving average
    plt.plot(scores, label="Score per Game")
    plt.plot(mean_scores, label="Average Score")

    # Keep y-axis starting from 0
    plt.ylim(ymin=0)

    # Annotate the latest values on the plot
    if len(scores) > 0:
        plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    if len(mean_scores) > 0:
        plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))

    # Add legend for clarity
    plt.legend()

    # Render the updated plot
    plt.show(block=False)
    plt.pause(0.1)   # small pause to allow GUI update
