import matplotlib.pyplot as plt
import argparse


def plot_optimization_trend(file_path):
    try:
        with open(file_path, 'r') as file:
            execution_times = [float(line.strip()) for line in file.readlines()]

        iterations = range(1, len(execution_times) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(iterations, execution_times, marker='o')
        plt.title('Optimization Trend')
        plt.xlabel('Iteration')
        plt.ylabel('Load Execution Time (s)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    except FileNotFoundError:
        print("Error: The file was not found.")
    except ValueError:
        print("Error: There was an issue converting a line to a float.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot optimization trend from a file.')
    parser.add_argument('--input', type=str, required=True, help='Path to the input file.')
    args = parser.parse_args()
    plot_optimization_trend(args.input)