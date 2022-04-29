# Python library
Have a look at the example notebook to see how the library is used.

## List of dictionary keys
- timestamp: UNIX timestamp in ms when packet was received
- packetNumber: Sent with BLE packet from wristband, should count up 1 per packet and wrap around 1 byte
- lostPackets: Number of lost packets since last received packet
- note: String of user note during recording
- ax: Wristband accelerometer X-axis [g]
- ay: Wristband accelerometer Y-axis [g]
- az: Wristband accelerometer Z-axis [g]
- gx: Wristband gyroscope X-axis [deg/s]
- gy: Wristband gyroscope Y-axis [deg/s]
- gz: Wristband gyroscope Z-axis [deg/s]
- mx: Wristband magnetometer X-axis [µT]
- my: Wristband magnetometer Y-axis [µT]
- mz: Wristband magnetometer Z-axis [µT]
- temperature: Wristband IMU die temperature [°C]
- longitude: Phone GPS longitude [degrees]
- latitude: Phone GPS latitude [degrees]
- altitude: Phone GPS altitude [m]
- bearing: Phone GPS bearing [degreees]
- speet: Phone GPS speed [m/s]
- phone_ax: Phone accelerometer X-axis [m/s^2]
- phone_ay: Phone accelerometer Y-axis [m/s^2]
- phone_az: Phone accelerometer Z-axis [m/s^2]
- phone_gx: Phone gyroscope X-axis [rad/s]
- phone_gy: Phone gyroscope Y-axis [rad/s]
- phone_gz: Phone gyroscope Z-axis [rad/s]
- phone_mx: Phone magnetometer X-axis [µT]
- phone_my: Phone magnetometer Y-axis [µT]
- phone_mz: Phone magnetometer Z-axis [µT]
- phone_gravx: Phone gravity X-axis [m/s^2]
- phone_gravy: Phone gravity Y-axis [m/s^2]
- phone_gravz: Phone gravity Z-axis [m/s^2]
- phone_lax: Phone linear acceleration X-axis [m/s^2]
- phone_lay: Phone linear acceleration Y-axis [m/s^2]
- phone_laz: Phone linear acceleration Z-axis [m/s^2]
- phone_rotx: Phone rotation vector X-axis 
- phone_roty: Phone rotation vector Y-axis
- phone_rotz: Phone rotation vector Z-axis
- phone_rotm: Phone rotation vector cos(θ/2)
- phone_magrotx: Phone geomagnetic rotation vector X-axis
- phone_magroty: Phone geomagnetic rotation vector Y-axis
- phone_magrotz: Phone geomagnetic rotation vector Z-axis
- phone_orientationx: Phone [orientation](https://developer.android.com/reference/android/hardware/Sensor#TYPE_ORIENTATION) azimuth [degrees]
- phone_orientationy: Phone orientation pitch [degrees]
- phone_orientationz: Phone orientation roll [degrees]
- phone_steps: Phone step counter (probably won't work)
- phone_temp: Phone temperature sensor [°C]
- phone_light: Phone light sensor [lx]
- phone_pressure: Phone pressure sensor [hPa or mbar]
- phone_humidity: Phone humidity sensor [%]



Find more info about the android sensors here:
- https://developer.android.com/guide/topics/sensors/sensors_overview
- https://developer.android.com/guide/topics/sensors/sensors_position
- https://developer.android.com/reference/android/location/Location
- https://developer.android.com/guide/topics/sensors/sensors_environment