import pygame
import numpy as np
import gym
from gym import spaces
import math
import time

# --COLORS--
RED = (255, 30, 70)
GREEN = (0,255,0)
# --######--

# --WINDOW--
window_width, window_height = 1000, 500 
# --######--

# Max Time of an iteration, defined by time module
MAX_TIME = 5


# Ball starting coordinates
BALL_START_X = 150
BALL_START_Y = 400




class CustomEnv(gym.Env):
    def __init__(self,env_config={}):
        self.observation_space = spaces.Box(low=0, high=1., shape=(2,))
        self.action_space = spaces.MultiDiscrete([360, 50, 2]) #list of size 3, each entry is a discrete number between 0 and entry (exclusive)
        
        # Define size of ball
        self.ball_radius = 15
        
        # Set to default starting coordinates of ball
        self.ball_x = BALL_START_X 
        self.ball_y = BALL_START_Y
        
        # For ball collision detection 
        self.ball_tuple = (self.ball_x, self.ball_y)
        self.ball_tuple_left = (self.ball_x - self.ball_radius, self.ball_y)
        self.ball_tuple_right = (self.ball_x + self.ball_radius, self.ball_y)
        self.ball_tuple_top = (self.ball_x, self.ball_y - self.ball_radius)
        self.ball_tuple_bottom = (self.ball_x, self.ball_y + self.ball_radius)
        
        
        # Game objects
        self.basket_left = pygame.Rect(800, 250, 5, 40)
        self.basket_right = pygame.Rect(840, 250, 5, 40)
        self.backboard = pygame.Rect(850, 190, 10, 60)
        self.buckets = pygame.Rect(810, 255, 25, 30)
        
        # Velocity control
        self.ball_velocityX = 0
        self.ball_velocityY = 0
        
        # Flag to control whether ball is thrown, and whether basket is made 
        self.thrown_flag = False
        self.buckets_flag = False
        
        # To display throw choice
        self.chosen_angle = 0
        self.chosen_velocity = 0
        
        # For reward score calculation
        self.net_distance_to_basket = 0
        
        # To control iteration length
        self.start_time = time.time() 

    def init_render(self):
        import pygame
        pygame.init()
        self.window = pygame.display.set_mode((window_width, window_height))
        self.clock = pygame.time.Clock()

    def reset(self):
        # Reset the environment to initial state
        self.ball_x = BALL_START_X
        self.ball_y = BALL_START_Y
        
        self.ball_velocityX = 0
        self.ball_velocityY = 0
        
        self.thrown_flag = False
        self.buckets_flag = False
        
        self.chosen_angle = 0
        self.chosen_velocity = 0
        
        self.net_distance_to_basket = 0
        
        self.start_time = time.time()
        
        
        # To report observation data
        buckets = 0
        if self.buckets_flag:
            buckets = 1
        return [np.interp(self.net_distance_to_basket, [0,300], [0,1]), buckets]

    def step(self, action=np.zeros((2),dtype=np.float)):
        '''
        Action space is a discrete space, uniform distribution of positive integers
        action[0] = angle choice, action[1] = velocity choice, action[2] = throw or not (1 for throw)
        '''
        
        # Only throw if a throw stimulus is inputted and the ball hasn't been thrown yet
        if action[2] == 1 and self.thrown_flag == False:
            
            raw_angle_choice = action[0] * -1
            raw_velocity_choice = action[1]
            
            # Convert to python-friendly units, enforce positive velocity
            angle_choice = math.radians(raw_angle_choice)
            velocity_choice = abs(raw_velocity_choice)
        
            # Trigonometry to get X and Y component vectors given angle choice
            self.ball_velocityX = velocity_choice * math.cos(angle_choice)
            self.ball_velocityY = velocity_choice * math.sin(angle_choice)
            
            # Update instance variables to store choice
            self.chosen_angle = raw_angle_choice * -1
            self.chosen_velocity = velocity_choice
            
            # Update flag to prevent re-throw
            self.thrown_flag = True
            
        
        # Physics checks: only do once ball is thrown
        if self.thrown_flag: 
            # "Air Resistance": reduce x velocity towards 0
            if self.ball_velocityX > 0:
                self.ball_velocityX -= 0.1          
            elif self.ball_velocityX < 0:
                self.ball_velocityX += 0.1 
            
            # "Gravity": pull ball towards bottom of screen
            self.ball_velocityY += 1           
        
        
        # -- Update ball position according to velocity -- 
        self.ball_x += self.ball_velocityX
        self.ball_y += self.ball_velocityY
        # ------------------------------------------------
        
        
        # -- Update ball's coordinate tuples --
        self.ball_tuple = (self.ball_x, self.ball_y)
        self.ball_tuple_left = (self.ball_x - self.ball_radius, self.ball_y)
        self.ball_tuple_right = (self.ball_x + self.ball_radius, self.ball_y)
        self.ball_tuple_top = (self.ball_x, self.ball_y - self.ball_radius)
        self.ball_tuple_bottom = (self.ball_x, self.ball_y + self.ball_radius)
        # -------------------------------------
        
          # -- Bounce collision with window -- 
        if self.ball_y > window_height:
            self.ball_velocityY *= -1 
            
        if self.ball_x > window_width or self.ball_x < 0:
            self.ball_velocityX *= -1
        # ----------------------------------
        
        # -- Check bounce collisions with basket and backboard -- 
        
        collide_flag_basket1 = ( self.basket_left.collidepoint(self.ball_tuple) or self.basket_left.collidepoint(self.ball_tuple_left)
                                or self.basket_left.collidepoint(self.ball_tuple_right) or self.basket_left.collidepoint(self.ball_tuple_top) 
                                or self.basket_left.collidepoint(self.ball_tuple_bottom) )
        collide_flag_basket2 = ( self.basket_right.collidepoint(self.ball_tuple) or self.basket_right.collidepoint(self.ball_tuple_bottom) 
                                or self.basket_right.collidepoint(self.ball_tuple_top) or self.basket_right.collidepoint(self.ball_tuple_left)
                                or self.basket_right.collidepoint(self.ball_tuple_right) )
        collide_flag_backboard = ( self.backboard.collidepoint(self.ball_tuple_bottom) or self.backboard.collidepoint(self.ball_tuple_top) 
                                  or self.backboard.collidepoint(self.ball_tuple_left) or self.backboard.collidepoint(self.ball_tuple_right) 
                                  or self.backboard.collidepoint(self.ball_tuple) )
        
        '''
        ### Version of collision detection only checking center
        collide_flag_basket1 = self.basket_left.collidepoint(self.ball_tuple)
        collide_flag_basket2 = self.basket_right.collidepoint(self.ball_tuple)
        collide_flag_backboard = self.backboard.collidepoint(self.ball_tuple)
        '''
    
        if collide_flag_backboard or collide_flag_basket1 or collide_flag_basket2:            
            self.ball_velocityX *= -1
        # --------------------------------------------------------
        
        
        # -- Check if it's a bucket --
        collide_flag_bucket = self.buckets.collidepoint(self.ball_tuple) or self.buckets.collidepoint(self.ball_tuple_left) or self.buckets.collidepoint(self.ball_tuple_right)
        
        if collide_flag_bucket:
            #print("Congrats!")
            self.buckets_flag = True
        # -----------------------------
        
        # -- Observation data calculations -- 
        x1,y1 = self.ball_tuple
        x2,y2 = self.buckets.center
        dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        dist_normalized = np.interp(dist, [0, window_height + window_width], [0, 1])
        
        self.net_distance_to_basket += dist_normalized
        
        # -- Reward calculation -- 
        reward_val = 0
    
        buckets = 0
        if self.buckets_flag:
            buckets = 1
    
        reward_val = (buckets) * 50 - (dist_normalized * 10)
        
        # -- Stop after x seconds, if no bucket has been made. Stop time defined globally --
        done_flag = False
        current_time = time.time()
        if current_time - self.start_time >= MAX_TIME:
            done_flag = True
            
        
        observation, reward, done, info = [dist_normalized, buckets], reward_val, (self.buckets_flag or done_flag), {}
        
        
        return observation, reward, done, info
    
    
    
    def render(self):
         
        
        self.window.fill((0,0,0))
        
        # -- Draw all objects --
        pygame.draw.rect(self.window, RED, self.basket_left)
        pygame.draw.rect(self.window, RED, self.basket_right)
        pygame.draw.rect(self.window, RED, self.backboard)
        pygame.draw.rect(self.window, GREEN, self.buckets)
        pygame.draw.circle(self.window, RED, (self.ball_x, self.ball_y), self.ball_radius)
        
        
        # -- Render telemetry text -- 
        font_obj=pygame.font.Font("C:\Windows\Fonts\Arial.ttf",20) 
        text = f"Chosen angle: {self.chosen_angle}°      Chosen velocity: {self.chosen_velocity}      Bucket: {self.buckets_flag}"
        text_obj=font_obj.render(text,True,GREEN) 
        self.window.blit(text_obj,(50,50)) 
      
      
        pygame.display.update()
        
        
    def pressed_to_action(self, inputnumber):
        '''
        User input function
        '''
        velocity_choice = 0
        angle_choice = 0
        throw_variable = 0
        
        if inputnumber == 0:
            pass
        if inputnumber == 1:  # 
            velocity_choice = 33
            angle_choice = 60 
            throw_variable = 1
        return np.array([angle_choice, velocity_choice, throw_variable])


if __name__ == "__main__":
    environment = CustomEnv()
    environment.init_render()
    run = True
    while run:
        # set game speed to 30 fps
        environment.clock.tick(30)
        # ─── CONTROLS ───────────────────────────────────────────────────────────────────
        # end while-loop when window is closed
        get_event = pygame.event.get()
        
        get_pressed = 0
        
        for event in get_event:
            if event.type == pygame.QUIT:
                run = False
                
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    get_pressed = 1
                        
        
        action = environment.pressed_to_action(inputnumber=get_pressed)
        # calculate one step
        environment.step(action)
        # render current state
        environment.render()
    pygame.quit()