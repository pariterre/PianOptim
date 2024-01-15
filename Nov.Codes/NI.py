
# Scenario Pressed
n_shooting_1 = (30, 7, 9, 10, 10)
phase_time_1 = (0.3, 0.044, 0.051, 0.15, 0.15)

# Scenario Struck
n_shooting_2 = (30, 6, 9, 10, 10)
phase_time_2 = (0.3, 0.027, 0.058, 0.15, 0.15)

# Calculations for Scenario 1
total_duration_1 = sum(phase_time_1)
total_frames_1 = sum(n_shooting_1)
frame_rate_1 = total_frames_1 / total_duration_1

# Calculations for Scenario 2
total_duration_2 = sum(phase_time_2)
total_frames_2 = sum(n_shooting_2)
frame_rate_2 = total_frames_2 / total_duration_2

print(total_duration_1, frame_rate_1)
print(total_duration_2, frame_rate_2)