from custom_environment import CustomEnv
import pygame
import csv

ITERATIONS = 25000 # Iteration total
FPS_SPEED = 300 # Speed of simulation, in FPS
DATABASE_PATH = r"C:\Documents_Austin\Spring 2022\cs181\FInal_project\data"
TRIAL_NUM = 1 # For easy naming af csv output


# Instantiate pygame environment
environment = CustomEnv()
environment.init_render()

# Run simulation number of times defined above
for i in range(ITERATIONS):
    
    environment.reset()
    done = False
    
    # Run pygame simluation until one "run" complete (either basket made or max time elasped)
    while not done:
        # set game speed to x fps, defined globally
        environment.clock.tick(FPS_SPEED)
        # ─── CONTROLS ───────────────────────────────────────────────────────────────────
        # end while-loop when window is closed
        get_event = pygame.event.get()
        
        get_pressed = 0
        
        for event in get_event:
            if event.type == pygame.QUIT:
                done = True
        
        # sample a random action from action space
        action = environment.action_space.sample()
        # calculate one step
        observation, reward, bucket_flag, info = environment.step(action)
        
        # check finished state
        done = bucket_flag
        
        # render current state
        environment.render()
        

    # Extract metrics
    angle, velocity = environment.get_throw()
    net_dist = environment.get_net_distance()
    bucket_made = environment.get_outcome()
    print(f"Angle: {angle}, Velocity: {velocity}")
    print(f"Net Distance from Basket: {net_dist}")
    print(f"Bucket made: {bucket_made}")
    
    # Write to database
    with open(f"{DATABASE_PATH}\{str(TRIAL_NUM)}.txt", "a+", newline='') as f:
        # Write header
        writer = csv.writer(f)
        if i == 0:
            writer.writerow(["angle", "velocity", "netDistanceFromBucket", "outcome"])
        
        data_row = [angle, velocity, net_dist, bucket_made]
        
        writer.writerow(data_row)
        

        
        