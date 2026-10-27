G = 9.81

BOUNCE_REWARD = 1.0
BAND_REWARD = 1.0
APEX_BAND = (0.20, 0.40)
MIN_APEX = 0.05


def apex_height(up_speed):
    return max(up_speed, 0.0) ** 2 / (2 * G)


def compute_reward(data, bounced):
    if not bounced:
        return 0.0
    apex = apex_height(data.joint("ball_free").qvel[2])
    if apex < MIN_APEX:
        return 0.0
    reward = BOUNCE_REWARD
    if APEX_BAND[0] <= apex <= APEX_BAND[1]:
        reward += BAND_REWARD
    return reward
