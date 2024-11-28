import numpy as np
import matplotlib.pyplot as plt
#Declare constant
to = 0.5
omega = 24
epsilon = np.exp(-to/omega)
A = 1/15
theta = 1

# Define parameters
L, U = 28.5, 33.5  # Range for ETIS
alpha1 = 0.5
alpha2 = 0.9
beta = 0.5

#Define Temperature and Humidity Information
T_tout = [23.4, 23.3, 23.2,24.85 , 21.5, 22.95, 21.4, 21.25, 21.1, 21.2, 21.3, 21.9, 20.5, 21.3, 23.1, 24.0, 24.9, 25.4, 25.9, 27.3, 28.7, 29.5, 30.3, 30.65, 31, 31.95, 32.9, 32.55, 32.2, 32.05, 31.9, 30.75, 29.6, 28.65, 27.7, 27.3, 27.1, 26.9, 26.3, 25.7, 24.95, 24.2, 24.5 ,24.8, 24.1, 22.6, 23.4, 22.3]
RH = [64, 64, 64, 63.5, 63, 63, 63, 63, 63, 62.5, 62, 62.5, 63, 76.5, 100, 96, 92, 85.5, 79, 71.5, 64, 55.5, 47, 39, 31, 30, 29, 28, 27, 26.5, 26, 28, 30, 33.5, 37, 37.5, 38, 38, 38, 39.5, 41, 43.5, 46, 46.5, 47, 46, 45, 44.5]
u = [0.775, 0.8, 0.825, 0.8306, 0.8361, 0.8514, 0.8667, 0.9306, 0.9944, 1.0292, 1.0639, 1.0585, 1.0532, 0.8475, 0.6417, 0.7862, 0.9306, 1.0681, 1.2056, 1.2806, 1.3556, 1.3654, 1.3752, 1.3182, 1.2611, 1.2222, 1.1833, 1.1722, 1.1611, 1.1903, 1.2194, 1.2250, 1.2306, 1.2042, 1.1778, 1.1473, 1.1167,  1.1862, 1.2556, 1.2348, 1.2139, 1.0611, 0.9083, 0.8028, 0.6972, 0.7278, 0.7583, 0.7642]
e_t = np.zeros(48)
T_Tin = np.zeros(48)
ETIS = np.zeros(48)
#T_tout = T_tout - np.ones(48)*10

# Define functions T_Tin, ETIS
def e_T(T_tin, T_tPre, T_tout):
    return (A/theta)*((T_tin-epsilon*T_tPre)/(1-epsilon)-T_tout)

def ETISFunc(T_Tin, RH_t, u_t):
    return T_Tin + 0.0006*(RH_t - 50)*T_Tin -0.3132*np.power(u_t,0.6827)*(38-T_Tin)-4.79*(1.0086*38-T_Tin)+4.8957*1e-8*(np.power(38+273.15,4)-np.power(T_Tin+273.15,4))

#Check constrant

# Objective function
def objective_function(T_Tin):
    smoothness = 0
    penalties = 0
    total_sum = np.sum(e_t)
    
    # Initialize f(x_1)
    T_Tin[0] = T_tout[0]
    ETIS[0] = ETISFunc(T_Tin[0], RH[0], u[0])
    
    # Iterate over the rest of x values
    for i in range(1, len(T_Tin)):
        e_t[i-1] = e_T(T_Tin[i], T_Tin[i-1], T_tout[i-1])
        if (e_t[i-1] < 0.5) :
            e_t[i-1] = 0
            T_Tin[i] = 0.8*T_Tin[i-1]+(1-0.8)*T_tout[i-1]
        else:
            if (e_t[i-1] > 20) :
                e_t[i-1] = 20
                T_Tin[i] = epsilon*T_Tin[i-1]+(1-epsilon)*(T_tout[i-1] +e_t[i-1]*theta/A) 
        ETIS[i] = ETISFunc(T_Tin[i], RH[i], u[i])  
        # Smoothness (minimize differences between consecutive g(f(x)) values)
        smoothness += (ETIS[i] - ETIS[i-1]) ** 2
        
        # Penalty if g(f(x)) is out of range
        if not (L <= ETIS[i] <= U):
            penalties += 1
    
    # Combine the objectives
    total_objective = alpha1 * np.sqrt(smoothness) + alpha2 * penalties + beta * total_sum
    return total_objective

# PSO parameters
num_particles = 1000
num_dimensions = 48  # Set according to the problem's dimensions
num_iterations = 100
inertia = 0.5
cognitive_param = 1.25
social_param = 1.75

# Initialize particles' positions and velocities
positions = np.random.uniform(0, 38, (num_particles, num_dimensions))
velocities = np.random.uniform(-1, 1, (num_particles, num_dimensions))
personal_best_positions = positions.copy()
personal_best_scores = np.array([objective_function(p) for p in positions])

# Initialize global best
global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
global_best_score = np.min(personal_best_scores)

# PSO main loop
for i in range(num_iterations):
    for j in range(num_particles):
        # Update velocities
        r1, r2 = np.random.rand(), np.random.rand()
        velocities[j] = (
            inertia * velocities[j]
            + cognitive_param * r1 * (personal_best_positions[j] - positions[j])
            + social_param * r2 * (global_best_position - positions[j])
        )

        # Update positions
        positions[j] += velocities[j]
        
        # Update personal bests
        score = objective_function(positions[j])
        if score < personal_best_scores[j]:
            personal_best_scores[j] = score
            personal_best_positions[j] = positions[j]

    # Update global best
    best_particle_idx = np.argmin(personal_best_scores)
    if personal_best_scores[best_particle_idx] < global_best_score:
        global_best_score = personal_best_scores[best_particle_idx]
        global_best_position = personal_best_positions[best_particle_idx]

    #print(f"Iteration {i+1}/{num_iterations}, Best Score: {global_best_score}")

# Result
print("Optimal solution:", global_best_position)
print("Objective function value:", global_best_score)
total = np.sum(global_best_position)
#plot value
fig, axs = plt.subplots(6, 1, figsize=(10, 10))

# Plotting each dataset in a separate subplot
axs[4].plot(e_t, color='b', label='Data 1')
axs[4].set_title("Energy")
axs[4].grid(True)

axs[5].plot(ETIS, color='r', label='Data 2')
axs[5].set_title("ETIS")
axs[5].grid(True)

axs[0].plot(T_tout, color='g', label='Data 3')
axs[0].set_title("T_out")
axs[0].grid(True)

axs[1].plot(RH, color='g', label='Data 3')
axs[1].set_title("RH")
axs[1].grid(True)

axs[3].plot(global_best_position, color='g', label='Data 3')
axs[3].set_title("T_in")
axs[3].grid(True)

axs[2].plot(u, color='g', label='Data 3')
axs[2].set_title("Air Velocity")
axs[2].grid(True)

plt.tight_layout()
plt.show()