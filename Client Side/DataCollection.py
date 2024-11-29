print("\033cStarting ...\n")  # To clear terminal

import time
import ximu3
import math
import cv2
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import threading
import sys
from pupil_labs.realtime_api.simple import discover_one_device

NEW_CAM = False


# Initialize sensor values
gyr_x_1 = gyr_y_1 = gyr_z_1 = 0
acc_x_1 = acc_y_1 = acc_z_1 = 0
gyr_x_2 = gyr_y_2 = gyr_z_2 = 0
acc_x_2 = acc_y_2 = acc_z_2 = 0

LINE_UP = '\033[1A'
LINE_CLEAR = '\x1b[2K'

# Connection class
class Connection:
    def __init__(self, connection_info):
        self.__connection = ximu3.Connection(connection_info)

        if self.__connection.open() != ximu3.RESULT_OK:
            raise Exception("Unable to open connection " + connection_info.to_string())

        ping_response = self.__connection.ping()

        if ping_response.result != ximu3.RESULT_OK:
            raise Exception("Ping failed for " + connection_info.to_string())

        self.__prefix = ping_response.serial_number
        self.__connection.add_inertial_callback(self.__inertial_callback)

    def close(self):
        self.__connection.close()

    def send_command(self, key, value=None):
        if value is None:
            value = "null"
        elif type(value) is bool:
            value = str(value).lower()
        elif type(value) is str:
            value = "\"" + value + "\""
        else:
            value = str(value)

        command = "{\"" + key + "\":" + value + "}"

        responses = self.__connection.send_commands([command], 2, 500)

        if not responses:
            raise Exception("Unable to confirm command " + command + " for " + self.__connection.get_info().to_string())
        else:
            print(self.__prefix + " " + responses[0])

    def __inertial_callback(self, message):
        global gyr_x_1, gyr_y_1, gyr_z_1
        global acc_x_1, acc_y_1, acc_z_1
        global gyr_x_2, gyr_y_2, gyr_z_2
        global acc_x_2, acc_y_2, acc_z_2
        if self.__prefix == '65577B49':
            gyr_x_1 = message.gyroscope_x
            gyr_y_1 = message.gyroscope_y
            gyr_z_1 = message.gyroscope_z
            acc_x_1 = message.accelerometer_x
            acc_y_1 = message.accelerometer_y
            acc_z_1 = message.accelerometer_z
        elif self.__prefix == '655782F7':
            gyr_x_2 = message.gyroscope_x
            gyr_y_2 = message.gyroscope_y
            gyr_z_2 = message.gyroscope_z
            acc_x_2 = message.accelerometer_x
            acc_y_2 = message.accelerometer_y
            acc_z_2 = message.accelerometer_z

# Establish connections
print("Checking connection to IMU ...")
while True :
    try :
        connections = [Connection(m.to_udp_connection_info()) for m in ximu3.NetworkAnnouncement().get_messages_after_short_delay()]
        break
    except AssertionError:
        pass
if not connections:
    print(LINE_UP, end=LINE_CLEAR)
    sys.exit("No UDP connections to IMUs")
print(LINE_UP, end=LINE_CLEAR)
print('Connected to IMUs')

# Video capture setup
print("Checking camera ...")
# Look for devices. Returns as soon as it has found the first device.
if NEW_CAM :
    device = discover_one_device(max_search_duration_seconds=10)
    if device is None:
        sys.exit("No device found.")
    print(LINE_UP, end=LINE_CLEAR)
    print(f"Connected to {device}")

else :
    cap = cv2.VideoCapture(0) # Set to 0 to activate the camera
    cap.set(cv2.CAP_PROP_FPS, 30)

Date = datetime.now().strftime("%Hh %M - %d %m %Y")
path = os.path.join('Data',Date)

os.makedirs(os.path.join(path,'images'))


# CSV file setup
csv_file = open(os.path.join(path,'imu_data.csv'), mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Timestamp', 'Gyr_X_1', 'Gyr_Y_1', 'Gyr_Z_1', 'Acc_X_1', 'Acc_Y_1', 'Acc_Z_1',
                     'Gyr_X_2', 'Gyr_Y_2', 'Gyr_Z_2', 'Acc_X_2', 'Acc_Y_2', 'Acc_Z_2'])

frame_count = 0

# Set up plotting
fig, ((axis_acc_1, axis_gyr_1), (axis_acc_2, axis_gyr_2)) = plt.subplots(2, 2, figsize=(10, 8))
axis_acc_1.set_title("IMU1 Accelerometer")
axis_gyr_1.set_title("IMU1 Gyroscope")
axis_acc_2.set_title("IMU2 Accelerometer")
axis_gyr_2.set_title("IMU2 Gyroscope")

x_len = 200  # Number of points to display

# Initialize lists to store data
xs = list(range(0, x_len))
acc_x_1_vals = [0] * x_len
acc_y_1_vals = [0] * x_len
acc_z_1_vals = [0] * x_len
gyr_x_1_vals = [0] * x_len
gyr_y_1_vals = [0] * x_len
gyr_z_1_vals = [0] * x_len
acc_x_2_vals = [0] * x_len
acc_y_2_vals = [0] * x_len
acc_z_2_vals = [0] * x_len
gyr_x_2_vals = [0] * x_len
gyr_y_2_vals = [0] * x_len
gyr_z_2_vals = [0] * x_len

# Create lines for each IMU's data
acc_x_1_line, = axis_acc_1.plot(xs, acc_x_1_vals, label='Acc X1')
acc_y_1_line, = axis_acc_1.plot(xs, acc_y_1_vals, label='Acc Y1')
acc_z_1_line, = axis_acc_1.plot(xs, acc_z_1_vals, label='Acc Z1')
gyr_x_1_line, = axis_gyr_1.plot(xs, gyr_x_1_vals, label='Gyr X1')
gyr_y_1_line, = axis_gyr_1.plot(xs, gyr_y_1_vals, label='Gyr Y1')
gyr_z_1_line, = axis_gyr_1.plot(xs, gyr_z_1_vals, label='Gyr Z1')
acc_x_2_line, = axis_acc_2.plot(xs, acc_x_2_vals, label='Acc X2')
acc_y_2_line, = axis_acc_2.plot(xs, acc_y_2_vals, label='Acc Y2')
acc_z_2_line, = axis_acc_2.plot(xs, acc_z_2_vals, label='Acc Z2')
gyr_x_2_line, = axis_gyr_2.plot(xs, gyr_x_2_vals, label='Gyr X2')
gyr_y_2_line, = axis_gyr_2.plot(xs, gyr_y_2_vals, label='Gyr Y2')
gyr_z_2_line, = axis_gyr_2.plot(xs, gyr_z_2_vals, label='Gyr Z2')

# Format plots
for ax in [axis_acc_1, axis_gyr_1, axis_acc_2, axis_gyr_2]:
    ax.legend(loc='upper right')


# Animation function
def animate(i, acc_x_1_vals, acc_y_1_vals, acc_z_1_vals, gyr_x_1_vals, gyr_y_1_vals, gyr_z_1_vals,
            acc_x_2_vals, acc_y_2_vals, acc_z_2_vals, gyr_x_2_vals, gyr_y_2_vals, gyr_z_2_vals):
    # Add new values
    acc_x_1_vals.append(acc_x_1)
    acc_y_1_vals.append(acc_y_1)
    acc_z_1_vals.append(acc_z_1)
    gyr_x_1_vals.append(gyr_x_1)
    gyr_y_1_vals.append(gyr_y_1)
    gyr_z_1_vals.append(gyr_z_1)
    acc_x_2_vals.append(acc_x_2)
    acc_y_2_vals.append(acc_y_2)
    acc_z_2_vals.append(acc_z_2)
    gyr_x_2_vals.append(gyr_x_2)
    gyr_y_2_vals.append(gyr_y_2)
    gyr_z_2_vals.append(gyr_z_2)

    # Limit lists to x_len items
    acc_x_1_vals = acc_x_1_vals[-x_len:]
    acc_y_1_vals = acc_y_1_vals[-x_len:]
    acc_z_1_vals = acc_z_1_vals[-x_len:]
    gyr_x_1_vals = gyr_x_1_vals[-x_len:]
    gyr_y_1_vals = gyr_y_1_vals[-x_len:]
    gyr_z_1_vals = gyr_z_1_vals[-x_len:]
    acc_x_2_vals = acc_x_2_vals[-x_len:]
    acc_y_2_vals = acc_y_2_vals[-x_len:]
    acc_z_2_vals = acc_z_2_vals[-x_len:]
    gyr_x_2_vals = gyr_x_2_vals[-x_len:]
    gyr_y_2_vals = gyr_y_2_vals[-x_len:]
    gyr_z_2_vals = gyr_z_2_vals[-x_len:]

    # Update lines with new values
    acc_x_1_line.set_ydata(acc_x_1_vals)
    acc_y_1_line.set_ydata(acc_y_1_vals)
    acc_z_1_line.set_ydata(acc_z_1_vals)
    gyr_x_1_line.set_ydata(gyr_x_1_vals)
    gyr_y_1_line.set_ydata(gyr_y_1_vals)
    gyr_z_1_line.set_ydata(gyr_z_1_vals)
    acc_x_2_line.set_ydata(acc_x_2_vals)
    acc_y_2_line.set_ydata(acc_y_2_vals)
    acc_z_2_line.set_ydata(acc_z_2_vals)
    gyr_x_2_line.set_ydata(gyr_x_2_vals)
    gyr_y_2_line.set_ydata(gyr_y_2_vals)
    gyr_z_2_line.set_ydata(gyr_z_2_vals)

    # Adjust y-axis range based on data
    axis_acc_1.set_ylim(min(acc_x_1_vals + acc_y_1_vals + acc_z_1_vals)-5, max(acc_x_1_vals + acc_y_1_vals + acc_z_1_vals)+5)
    axis_gyr_1.set_ylim(min(gyr_x_1_vals + gyr_y_1_vals + gyr_z_1_vals)-30, max(gyr_x_1_vals + gyr_y_1_vals + gyr_z_1_vals)+30)
    axis_acc_2.set_ylim(min(acc_x_2_vals + acc_y_2_vals + acc_z_2_vals)-5, max(acc_x_2_vals + acc_y_2_vals + acc_z_2_vals)+5)
    axis_gyr_2.set_ylim(min(gyr_x_2_vals + gyr_y_2_vals + gyr_z_2_vals)-30, max(gyr_x_2_vals + gyr_y_2_vals + gyr_z_2_vals)+30)

    return acc_x_1_line, acc_y_1_line, acc_z_1_line, gyr_x_1_line, gyr_y_1_line, gyr_z_1_line, \
        acc_x_2_line, acc_y_2_line, acc_z_2_line, gyr_x_2_line, gyr_y_2_line, gyr_z_2_line


# Set up plot to call animate() function periodically

ani = animation.FuncAnimation(
    fig,
    animate,
    fargs=(acc_x_1_vals, acc_y_1_vals, acc_z_1_vals, gyr_x_1_vals, gyr_y_1_vals, gyr_z_1_vals,
           acc_x_2_vals, acc_y_2_vals, acc_z_2_vals, gyr_x_2_vals, gyr_y_2_vals, gyr_z_2_vals),
    interval=50,
    #blit=True, # Comment to have the axis to update their values
    save_count=200  # 这个值可以根据需求调整 (This value can be adjusted as required)
)

# Show video and plot in real-time
plt.tight_layout()


def capture_and_plot():
    global frame_count
    start_time = time.time()
    while True:
        if NEW_CAM :
            bgr_pixels, frame_datetime = device.receive_scene_video_frame()
            frame = bgr_pixels # TODO Possible source of error, check conversion
        else :
            ret, frame = cap.read()
            if not ret: # If camera is unavailable :
                # Release resources
                cap.release()
                cv2.destroyAllWindows()
                csv_file.close()
                for connection in connections:
                    connection.close()
                print('\nCamera disconnected')
                raise KeyboardInterrupt

        # Ensure 30 FPS
        while time.time() - start_time < frame_count / 30:
            time.sleep(0.001)
        image_filename = f'{path}/images/frame_{frame_count}.jpg'
        cv2.imwrite(image_filename, frame)
        # Write IMU data to CSV with timestamp
        csv_writer.writerow([frame_count, gyr_x_1, gyr_y_1, gyr_z_1, acc_x_1, acc_y_1, acc_z_1,
                             gyr_x_2, gyr_y_2, gyr_z_2, acc_x_2, acc_y_2, acc_z_2])

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# Run the capture and plot in parallel
start = time.time()
capture_thread = threading.Thread(target=capture_and_plot)
capture_thread.start()

plt.show()

# Release resources
csv_file.close()
device.close()  # explicitly stop auto-update
cv2.destroyAllWindows()
for connection in connections:
    connection.close()
