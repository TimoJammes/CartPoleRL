import pygame
import numpy as np
from enum import Enum

from .config import DisplayConfig, PhysicsConfig
from .env import CartPoleEnv
from .actions import Action

class InputCommand(Enum):
    QUIT = 0
    HEADLESS_SWITCH = 1

class PygameUI:

    def __init__(self, cfg: DisplayConfig, physics_cfg: PhysicsConfig):
        pygame.init()

        self.cfg = cfg
        # WIDTH, HEIGHT = cfg.window_width, cfg.window_height
        # AREA = (WIDTH, HEIGHT)
        self.screen = pygame.display.set_mode((self.cfg.window_width, self.cfg.window_height))
        self.clock = pygame.time.Clock()

        self.to_screen_scaling = (self.cfg.window_width - self.cfg.cart_width) / physics_cfg.track_length

        self.fps = round(1 / physics_cfg.dt)
        
        
        self.font = pygame.font.SysFont(None, 32)
        self.LEFT_TEXT = self.font.render("LEFT", False, pygame.Color("black")) 
        self.NONE_TEXT = self.font.render("NONE", False, pygame.Color("black")) 
        self.RIGHT_TEXT = self.font.render("RIGHT", False, pygame.Color("black")) 
        self.PAUSE_TEXT = self.font.render("RENDER PAUSED", False, pygame.Color("white"))
        self.EPISODE_TEXT = self.font.render("EPISODE: ", False, pygame.Color("black"))
        self.max_episode_num_text_width = self.font.render("999999", False, pygame.Color("black")).get_width()
        
        self.episode_num_background_rect = pygame.Rect(self.cfg.window_width-self.EPISODE_TEXT.get_width()-self.max_episode_num_text_width,
                                                       0,
                                                       self.EPISODE_TEXT.get_width()+self.max_episode_num_text_width,
                                                       self.EPISODE_TEXT.get_height())
            
    def process_events(self) -> set[InputCommand]:
        events = set()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                events.add(InputCommand.QUIT)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    events.add(InputCommand.HEADLESS_SWITCH)

        return events
    
    def tick(self):
        self.clock.tick(self.fps)

    def render(self, env: CartPoleEnv, action: Action, episode_num: int, do_render: bool):

        if not do_render:
            
            if episode_num % 50 == 0:
                pygame.draw.rect(self.screen, self.cfg.background_color,
                                self.episode_num_background_rect)
                self.display_episode_num(episode_num)
                pygame.display.flip()
            return
            
        self.draw_background()
        
        self.draw_cart(env)
        self.draw_track(env)
        self.draw_pole(env)

        self.display_action(action)
        self.display_episode_num(episode_num)
        
        pygame.display.flip()

    def draw_background(self):
        self.screen.fill(self.cfg.background_color)

    def draw_track(self, env: CartPoleEnv):
        screen_x_left, screen_y_left = self.world_to_screen(-env.physics_cfg.track_length//2, 0)
        screen_x_right, screen_y_right = self.world_to_screen(env.physics_cfg.track_length//2, 0)
        
        pygame.draw.line(self.screen, self.cfg.track_color, (screen_x_left, screen_y_left), (screen_x_right, screen_y_right), self.cfg.track_thickness)
    
    def draw_cart(self, env: CartPoleEnv):
        self.draw_cart_body(env)
        self.draw_cart_wheels(env)
            
    def draw_cart_body(self, env: CartPoleEnv):
        screen_x, screen_y = self.world_to_screen(env.state.x, 0)
        screen_y -= self.cfg.cart_height + self.cfg.track_thickness // 2 + self.cfg.cart_wheel_radius * 2
        
        pygame.draw.rect(self.screen, self.cfg.cart_body_color, pygame.Rect(screen_x-self.cfg.cart_width // 2, screen_y, self.cfg.cart_width, self.cfg.cart_height))
    
    def draw_cart_wheels(self, env: CartPoleEnv):
        _, screen_y = self.world_to_screen(-1, 0)
        screen_y -= self.cfg.track_thickness // 2 + self.cfg.cart_wheel_radius
        
        screen_x_wheel_1, _ = self.world_to_screen(env.state.x, -1)
        screen_x_wheel_2, _ = self.world_to_screen(env.state.x, -1)
        
        screen_x_wheel_1 -= self.cfg.cart_wheel_radius
        screen_x_wheel_2 += self.cfg.cart_wheel_radius
        
        pygame.draw.circle(self.screen, self.cfg.cart_wheel_color, (screen_x_wheel_1, screen_y), self.cfg.cart_wheel_radius)
        pygame.draw.circle(self.screen, self.cfg.cart_wheel_color, (screen_x_wheel_2, screen_y), self.cfg.cart_wheel_radius)
    
    def draw_pole(self, env: CartPoleEnv):

        theta = env.state.theta
        length = env.physics_cfg.pole_length

        # world coordinates: the pole is attached at the cart's center, at track height
        base_x, base_y = env.state.x, 0
        tip_x = base_x - length * np.sin(theta)
        tip_y = base_y - length * np.cos(theta)

        screen_base_x, screen_base_y = self.world_to_screen(base_x, base_y)
        screen_tip_x, screen_tip_y = self.world_to_screen(tip_x, tip_y)

        screen_base_y -= self.cfg.cart_wheel_radius * 2 + self.cfg.cart_height // 2
        screen_tip_y -= self.cfg.cart_wheel_radius * 2 + self.cfg.cart_height // 2
        
        pygame.draw.line(self.screen, self.cfg.pole_color, (screen_base_x, screen_base_y), (screen_tip_x, screen_tip_y), self.cfg.pole_width)

    def display_action(self, action: Action):

        text_x = 10

        text = self.LEFT_TEXT
        match action.name:
            case "NONE":
                text = self.NONE_TEXT
                text_x += self.LEFT_TEXT.get_width() + 10
            case "RIGHT":
                text = self.RIGHT_TEXT
                text_x += self.LEFT_TEXT.get_width() + self.NONE_TEXT.get_width() + 20

        self.screen.blit(text, (text_x, 0))

    def display_episode_num(self, episode_num: int):
        text = self.font.render(str(episode_num), False, pygame.Color("black"))
        
        self.screen.blit(self.EPISODE_TEXT, (self.cfg.window_width - text.get_width() - self.EPISODE_TEXT.get_width(), 0))
        self.screen.blit(text, (self.cfg.window_width - text.get_width(), 0))
        
    def draw_pause_screen(self):
        
        pygame.draw.rect(self.screen, pygame.Color("black"),
                         pygame.Rect(self.cfg.window_width // 2 - self.PAUSE_TEXT.get_width(),
                         self.cfg.window_height // 2 - self.PAUSE_TEXT.get_height(), self.PAUSE_TEXT.get_width() * 2,
                         self.PAUSE_TEXT.get_height() * 2))
        
        self.screen.blit(self.PAUSE_TEXT,
                          (self.cfg.window_width // 2 - self.PAUSE_TEXT.get_width() // 2, 
                           self.cfg.window_height // 2 - self.PAUSE_TEXT.get_height() //2))

        pygame.display.flip()
        
    def _rotate(self, p: tuple, center_p: tuple, theta: float) -> np.ndarray:
        p_array = np.array(p)
        center_p_array = np.array(center_p)

        cos = np.cos(theta)
        sin = np.sin(theta)

        rot = np.array([[cos, sin],
                        [-sin, cos]])
        
        return ((p_array-center_p_array) @ rot) + center_p_array

    def world_to_screen(self, world_x, world_y):
        return self.cfg.window_width // 2 + world_x * self.to_screen_scaling , self.cfg.track_y - world_y * self.to_screen_scaling 
    def quit(self):
        pygame.quit()