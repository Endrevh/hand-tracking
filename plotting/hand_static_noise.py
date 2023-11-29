import matplotlib.pyplot as plt
import numpy as np

def read_data(filename):
    times = []
    z1_values = []
    avg_z_values = []
    avg_z_values_no_tips = []  # List for averages without fingertips
    fingertip_indices = [4, 8, 12, 16, 20]
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            if len(parts) < 64:  # 1 time + 21*3 coordinates
                continue
            time = float(parts[0])
            z_coordinates = [float(parts[i]) for i in range(3, 64, 3)]
            z_coordinates_no_tips = [z_coordinates[i] for i in range(21) if i not in fingertip_indices]
            if 0.0 in z_coordinates: # If any of the coordinates are 0.0, skip this line
                continue
            z1 = z_coordinates[0]
            avg_z = np.mean(z_coordinates)
            avg_z_no_tips = np.mean(z_coordinates_no_tips)
            times.append(time)
            z1_values.append(z1)
            avg_z_values.append(avg_z)
            avg_z_values_no_tips.append(avg_z_no_tips)
    return times, z1_values, avg_z_values, avg_z_values_no_tips

def calculate_standard_deviation(values):
    return np.std(values)

def plot_data(times, z1_values, avg_z_values, avg_z_values_no_tips):
    times_seconds = [t/1000.0 for t in times]
    plt.plot(times_seconds, z1_values, label='Wrist landmark')
    plt.plot(times_seconds, avg_z_values, label='Average of all landmarks')
    plt.plot(times_seconds, avg_z_values_no_tips, label='Average of all landmarks excluding tips')
    plt.xlabel('Time [s]')
    plt.ylabel('Depth estimate [m]')
    #plt.title('Hand depth estimate, wrist vs. average of all landmarks')
    plt.title('Hand depth estimate, wrist vs. average of landmarks')
    plt.legend()
    plt.grid()
    plt.show()

# Main script
filename = '../data/static_noise_test.txt'  # Replace with your file name
times, z1_values, avg_z_values, avg_z_values_no_tips  = read_data(filename)

std_dev_z1 = calculate_standard_deviation(z1_values)
std_dev_avg_z = calculate_standard_deviation(avg_z_values)
std_dev_avg_z_no_tips = calculate_standard_deviation(avg_z_values_no_tips)  # Standard deviation excluding fingertips

print(f"Standard Deviation of wrist: {std_dev_z1}")
print(f"Standard Deviation of average z: {std_dev_avg_z}")
print(f"Standard Deviation of average z (excluding tips): {std_dev_avg_z_no_tips}")

plot_data(times, z1_values, avg_z_values, avg_z_values_no_tips)
