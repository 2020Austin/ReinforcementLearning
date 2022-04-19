from run_everything import CustomEnv
import pygame

ITERATIONS = 50 # Iteration total
FPS_SPEED = 60 # Speed of simulation

environment = CustomEnv()
environment.init_render()

for i in range(ITERATIONS):
    
    

    environment.reset()
    done = False
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
